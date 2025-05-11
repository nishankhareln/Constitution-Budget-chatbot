#  Nepali Constitution Chatbot 🇳🇵

An AI-powered chatbot that answers your questions based on the **Constitution of Nepal 2072** using OpenAI's GPT-4o and semantic search (FAISS + LangChain).

---

##  Features

-  Ask questions related to the Nepali Constitution
-  Reads and indexes content from an official PDF
-  Uses FAISS for efficient semantic search
-  Powered by OpenAI's GPT-4o (via LangChain)
-  Answers queries in natural, understandable language

---

## Setup Instructions

1. Clone this repository
git clone https://github.com/manishabhatt7/Nepali-Constitution-Chatbot.git
cd Nepali-Constitution-Chatbot

2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

3. Install all required Python packages
pip install -r requirements.txt

4. Add your OpenAI API key
Create a .env file in the project directory and add this line:

OPENAI_API_KEY=your-api-key-here
#Replace your-api-key-here with your actual OpenAI API key.

5. Download the Constitution PDF
Download the English version of the Constitution of Nepal from:
Constitution PDF -> (AG.gov.np)

Save it as:
Constitution-of-Nepal.pdf

6. Run the chatbot
python chatbot.py


System Flow & Architecture
1. Document Ingestion Pipeline
Input: PDF documents (Constitution-of-Nepal.pdf, budget.pdf)

Processing Steps:

Text Extraction:

Uses PyMuPDF to extract raw text from each page of the PDFs

Document Segmentation:

Splits text into chunks (1000 characters each, with 200-character overlap)

Tags each chunk with metadata (source document name)

Vector Embedding:

Converts text chunks into numerical vectors using all-MiniLM-L6-v2 embeddings

Vector Storage:

Stores embeddings in a FAISS vector database for fast similarity search

2. User Interaction Flow
Document Selection (Sidebar):

User selects which documents to query (Constitution, Budget, or both)

System dynamically filters the search space

Question Processing:

User enters a question in the chat interface

System converts question to embeddings and finds most relevant document chunks

AI Response Generation:

Relevant document chunks are sent to Groq's Llama3-70b model

Model synthesizes an answer citing sources

Response includes:

Generated answer

List of source documents used

Conversation Logging:

All Q&A pairs are stored with:

Original question

Generated answer

Timestamp

Documents consulted

Saved in session memory (can be exported to DB)

3. Technical Components
Component	Technology	Purpose
Document Loader	PyMuPDF	Extract text from PDFs
Text Splitter	LangChain Recursive Splitter	Create manageable chunks
Vector Database	FAISS	Store/search document embeddings
Embedding Model	HuggingFace MiniLM	Convert text to vectors
LLM	Groq (Llama3-70b)	Generate answers
UI Framework	Streamlit	Web interface
4. Key Features
Multi-Document Querying: Simultaneously search across constitution and budget

Source Attribution: Always shows which document provided the information

Session Memory: Remembers full conversation history

Dynamic Filtering: Instantly responds to document selection changes

Secure API Handling: Groq key managed via .env file only

5. Data Flow Diagram
User Question → [Streamlit UI] 
               ↓ 
[FAISS Vector DB] → Finds relevant chunks 
                     ↓ 
[Groq LLM] → Generates answer with sources
               ↓ 
[UI Display] ← Formatted response
               ↓ 
[Session State] → Logs conversation
6. Example Use Case
User selects both "Constitution" and "Budget"

Asks: "How does the budget implement constitutional education requirements?"

System:

Finds relevant education clauses in Constitution

Locates corresponding budget allocations

Generates comparative analysis

Returns answer with:

The Constitution mandates free education (Article 31)...
The 2024 budget allocates 15% to education...

📄 Sources: Constitution, Budget



                                                                                                                                                                                                                       
                                                                                                                                                                                                                     
                                                                                                                                                     
