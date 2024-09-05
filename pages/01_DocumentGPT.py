import streamlit as st
import time
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃"
)


class ChatCallbackHandler(BaseCallbackHandler):
    message=""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file (file):
    
    file_path = f"./.cache/files/{file.name}"
    folder_path = os.path.dirname(file_path)
    
    # No such file or directory issue 해결
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_content = file.read()
    
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter = splitter)
    embeddings = OpenAIEmbeddings()

    #chaching embeddings
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message,role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"],message["role"],save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)



st.title("DocumentGPT")


st.markdown("""
Welcome!

Use this Chatbot to ask question about the document that you upload!
First, Upload your files on the sidebar.
""")


with st.sidebar:
    file = st.file_uploader("Upload a .txt, .pdf or .docx file", type=["pdf", "txt", "docx"])
    st.markdown("## Please enter your OpenAI API Key")
    openai_key = st.text_input("OpenAI API Key", type="password")

    if openai_key:            
        ## Open AI llm settings
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()], api_key=openai_key)
        memory = ConversationBufferMemory(
            llm = llm,
            max_token_limit=150,
            return_messages=True,

        )
        ## Setting prompt with memory
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                """
                You are the helpfule assistant which answer the question using only the following context. 
                If you don't know the answer about the question, please just say you don't know.

                ---
                Context:{context}
                """),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ]
        )

def load_memory(_):
    return memory.load_memory_variables({})["history"]


if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context" : retriever | RunnableLambda(format_docs),
                "question" : RunnablePassthrough(),
                "history": load_memory
            }
            | prompt 
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
        
else:
    st.session_state["messages"] = []