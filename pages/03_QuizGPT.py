import os
import json
import streamlit as st

from operator import rshift
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.retrievers import WikipediaRetriever
from langchain.schema.runnable import RunnablePassthrough


st.set_page_config(
    page_title="Quiz GPT",
    page_icon="❓"
)

st.title("QuizGPT")


### function calling


function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}




def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
        
        Based on the user's chosen difficulty level — easy, or hard — generate a question differently. 
        If the difficulty is 'easy', create a simple and straightforward question that requires basic knowledge.
        For 'hard', generate the most complex and challenging questions that demands in-depth knowledge or problem-solving skills.
        
        ---
        Context: {context},
        Difficulty: {difficulty},
        """,
                )
            ]
        )


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    folder_path = os.path.dirname(file_path)

    # No such file or directory issue 해결
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs



@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    
    question_chain = question_prompt | llm
    response = question_chain.invoke({"context": format_docs(_docs), "difficulty":difficulty})
    response = response.additional_kwargs["function_call"]["arguments"]
    
    return json.loads(response)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

## session_state for submit button
if "disabled" not in st.session_state:
    st.session_state.disabled = False


with st.sidebar:
    st.markdown("## Please enter your OpenAI API Key")
    openai_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")
    difficulty = st.radio("Choose a difficulty level.", ("Hard", "Easy"))
    
    docs = None
    choice = st.selectbox(
        "Choose what you want to use for Quiz.",
        ("File", "Wikipedia Article")
    )
    
    if choice == "File":
        file = st.file_uploader(
            "Upload your file with .docx , .txt or .pdf extension",
            type=["pdf", "txt", "docx"]
        )

        if file:
            split_file(file)
    
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    
if openai_key:
    ## Open AI llm settings
    llm = ChatOpenAI(temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key=openai_key
    ).bind(
    function_call={"name": "create_quiz"},
    functions=[function],)


if not docs:
    st.markdown(
        """
    ### Welcome to QuizGPT.
    ##### I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
    
    ##### Please Enter Your OpenAI API Key and Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
    
else:
    if not openai_key:
        st.error("Please add your OpenAI API key to continue.")
        st.stop()

    response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
    score = 0
    total_quiz_count = len(response["questions"])
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio("Choose an answer",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                score += 1
            elif value is not None:
                st.error("Wrong!")
            st.markdown("---")
        button = st.form_submit_button()
        
        if button:
            if score == total_quiz_count:
                st.balloons()
                st.markdown("### 🎉 Congratulations! You got all the answers correct! 🎉")
            else:
                st.warning(f"You answered {score} out of {total_quiz_count} questions correctly! Keep up the great work, and don't stop challenging yourself! 🔥")


            
        