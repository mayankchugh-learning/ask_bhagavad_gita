import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate, LLMChain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain



# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google API key
GoogleGenerativeAIEmbeddings.api_key = api_key

# Function to load and split the text from a PDF file
def get_file_text():
    text = ""
    pdf_reader = PdfReader('data/Bhagavad-Gita As It Is.pdf')
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to load or create a vector store from the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """
You are Lord Krishna, and the user is like Arjuna to you. \n
You have to guide them as Krishna guided Arjuna on the battlefield \n
and provide them with the knowledge known as the Bhagavad Gita. \n
Similarly, when the user asks you a question, "{question}," \n
you must find the best answer for them based on the context and provide \n
them with the right guidance. Provide the answer based on the Gita. \n
The answer format should include 5-6 lines of response, a relevant shloka from the \n
Bhagavad Gita, and an example to help the user understand easily. Remember to translate the answer Hindi language.
    
Context:
{context}\n

Answer:
    """

    # Define the language model
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)

    # Create a prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Create the conversational chain
    return LLMChain(llm=model, prompt=prompt)

# Function to predict the next question
def predict_next_question(user_question):
    # Create a prompt template for predicting the next question
    prompt_template = """
    Based on the user's question: {user_question}, predict the next question the user might ask.
    """

    # Define the language model
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    
    # Create the prompt and LLM chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_question"])
    chain = LLMChain(llm=model, prompt=prompt)
    
    # Use the chain to predict the next question
    next_question = chain.run(user_question)
    
    return next_question

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the vector store
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform a similarity search and retrieve relevant documents as context
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Run the chain with the retrieved context and the user's question
    response = chain.run({"context": docs, "question": user_question})

    # Create a container to display the response
    with st.container():
        st.write("Reply: ", response)

    # Predict the next question
    next_question = predict_next_question(user_question)
    
    # Display the predicted next question in the sidebar
    st.sidebar.header("Predicted Next Question")
    st.sidebar.write(next_question)

# Main function
def main():
    st.set_page_config("Chat Lord Krishna")
    st.header("Chat with Bhagwan Shri Krishna 🌟")
    st.subheader("Seeking Guidance for Life's Questions")

    # User input
    import random
    ques =  [   'Am I good enough to achieve my goals?',
                'Am I capable of building and maintaining meaningful relationships?',
                'Do I have what it takes to succeed in life?']
    
    selected_ques = random.choice(ques)
    if st.button('Generate Random Question'):
        st.write(selected_ques)
        user_question = selected_ques

        with st.spinner('Wait for it...'):
            if user_question:
                user_input(user_question)
    
    user_question = st.text_input("Ask a Question to Bhagwan Krishna")
    if st.button("Submit"):
        with st.spinner('Wait for it...'):
            if user_question:
                user_input(user_question)

if __name__ == "__main__":
    main()
