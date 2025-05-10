import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# ------------------------ LOAD ENV ------------------------
load_dotenv()  # Load variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Securely loaded API key
PDF_PATH = "Constitution-of-Nepal.pdf"
# ----------------------------------------------------------


# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Step 2: Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


# Step 3: Create FAISS vector store from chunks
def create_vector_store(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# Step 4: Create a QA chain and answer a question
def ask_question(vector_store, question, openai_api_key):
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    answer = qa_chain.invoke(question)
    return answer


# Main function
def main():
    print("\n Loading Constitution of Nepal and building chatbot...")

    # Step 1
    text = extract_text_from_pdf(PDF_PATH)
    print(" Text extracted.")

    # Step 2
    chunks = split_text(text)
    print(f" Text split into {len(chunks)} chunks.")

    # Step 3
    vector_store = create_vector_store(chunks, OPENAI_API_KEY)
    print(" FAISS vector store created.")

    # Step 4: Ask in loop
    print("\nConstitution Chatbot is ready. Ask any question related to Nepali Constitution.\n(Type 'exit' to quit.)")
    while True:
        query = input("\n Your Question: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = ask_question(vector_store, query, OPENAI_API_KEY)
        print(f"\n Answer:\n{response}")


if __name__ == "__main__":
    main()
