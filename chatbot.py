import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(
    page_title="Nepal Government Documents Chatbot",
    page_icon="🇳🇵",
    layout="centered"
)

# Document configuration
DOCUMENTS = {
    "Constitution": "Constitution-of-Nepal.pdf",
    "Budget": "budget.pdf"
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = list(DOCUMENTS.keys())

# Check for Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found in .env file")
    st.stop()

# Sidebar for document selection
with st.sidebar:
    st.title("Document Selection")
    st.session_state.selected_docs = st.multiselect(
        "Select documents to query:",
        options=list(DOCUMENTS.keys()),
        default=st.session_state.selected_docs
    )
    st.divider()
    st.markdown("**About**")
    st.markdown("Chat with Nepal's official documents")
    st.markdown("- Powered by Groq's Llama3")
    st.markdown("- Stores all Q&A in vector database")

# Main app
st.title("🇳🇵 Nepal Government Documents Chatbot")
st.caption("Ask questions about selected documents")

# Process PDFs
@st.cache_resource
def process_documents(selected_docs):
    try:
        all_texts = []
        
        for doc_name in selected_docs:
            pdf_path = DOCUMENTS[doc_name]
            if not os.path.exists(pdf_path):
                st.error(f"File not found: {pdf_path}")
                continue
                
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                all_texts.append((doc_name, text))
        
        if not all_texts:
            st.error("No valid documents found")
            return None
            
        # Split all texts
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = []
        for doc_name, text in all_texts:
            doc_chunks = splitter.split_text(text)
            chunks.extend([
                Document(
                    page_content=chunk,
                    metadata={"source": doc_name}
                ) for chunk in doc_chunks
            ])
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)
        
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return None

# Initialize documents
if st.session_state.vector_store is None:
    with st.spinner(f"Loading {len(st.session_state.selected_docs)} document(s)..."):
        st.session_state.vector_store = process_documents(st.session_state.selected_docs)

# Store conversation
def store_conversation(prompt, answer):
    st.session_state.conversation_history.append({
        "question": prompt,
        "answer": answer,
        "documents": st.session_state.selected_docs,
        "timestamp": datetime.now().isoformat()
    })

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface
if prompt := st.chat_input("Ask about selected documents"):
    if not st.session_state.selected_docs:
        st.error("Please select at least one document")
        st.stop()
        
    if st.session_state.vector_store is None:
        st.error("Document processing failed")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama3-70b-8192",
                    temperature=0.3
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 5, "filter": {"source": st.session_state.selected_docs}}
                    ),
                    return_source_documents=True
                )
                
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                
                # Show document sources
                sources = {doc.metadata['source'] for doc in response['source_documents']}
                answer += f"\n\n📄 Sources: {', '.join(sources)}"
                
                # Store conversation
                store_conversation(prompt, answer)
                
                # Display
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Debug section
with st.expander("Session Info"):
    st.write("Selected Documents:", st.session_state.selected_docs)
    st.write("Conversation History:", st.session_state.conversation_history)