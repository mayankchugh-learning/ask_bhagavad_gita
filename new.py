import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_file_text():
    text = ""
    pdf_reader = PdfReader('data/Bhagavad-Gita As It Is.pdf')
    for page in pdf_reader.pages:
        text += page.extract_text()            

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """You are a lord krishna and the person or user is comming to you for finding
     the answer about thier life issue now you have read the question {question} and you have find the
    correct vage from the context and the provide the correct answer as helping him to solve the problem
    provide the small and short answer with 4-5 lines of information.
    provide the shlok lines which you find similar to the correct answer or problem
    with examples
    translate the answer in hindi 
    Context:\n {context}?\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    
    st.set_page_config("Chat Lord Krishna")
    st.header("Bhagvad Gita As It Is📖")
    st.subheader('Ask your question to Lord Krishan 🤔💭')

    user_question = st.text_input("Ask a Question to Bhagwan Krishna")

    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()