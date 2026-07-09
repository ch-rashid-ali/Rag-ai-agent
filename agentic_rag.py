import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Load Environment Variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Check if keys exist
if not GROQ_API_KEY or not PINECONE_API_KEY:
    st.error("Error: API Keys nahi mili! Kripya .env file ya Streamlit Secrets check karein.")
    st.stop()

# 2. Initialize Pinecone and Models
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-agent-index"

# Create Pinecone Index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, # Dimension for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768", temperature=0.2)

# 3. PDF Loading and Processing Function (Local Folder Se)
@st.cache_resource
def initialize_rag_system():
    # ---- APNI PDF FILE KA NAAM YAHAN BADLEIN ----
    pdf_filename = "Copy of Constitutional History of Pakistan.pdf" 
    
    if not os.path.exists(pdf_filename):
        st.error(f"Error: {pdf_filename} file folder me nahi mili! Kripya check karein.")
        st.stop()
        
    # Load and Split PDF
    loader = PyPDFLoader(pdf_filename)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Push Embeddings to Pinecone (Upsert)
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk.page_content).tolist()
        metadata = {"text": chunk.page_content, "page": chunk.metadata.get("page", 0)}
        index.upsert(vectors=[(f"vec_{i}", embedding, metadata)])
        
    return chunks

# Run Initialization
with st.spinner("PDF process ho rahi hai aur embeddings ban rahi hain... Kripya thoda intezar karein."):
    initialize_rag_system()

# 4. Custom Pinecone Retriever for LangChain
class PineconeRetriever:
    def __init__(self, index, embedding_model):
        self.index = index
        self.embedding_model = embedding_model
        
    def invoke(self, query):
        query_vector = self.embedding_model.encode(query).tolist()
        results = self.index.query(vector=query_vector, top_k=3, include_metadata=True)
        docs = []
        for match in results['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                docs.append(Document(page_content=match['metadata']['text'], metadata={"page": match['metadata'].get("page", 0)}))
        return docs

from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 5. Strict Prompt Engineering (Sirf PDF se jawab dene ke liye)
system_prompt = (
    "Aap ek expert AI assistant hain. Aapko sirf niche diye gaye context (PDF data) ka use karke user ke sawaal ka jawab dena hai.\n"
    "Agar jawab context me MAUJOOD NAHI HAI, toh saaf bol dein: 'Mujhe iska jawab di gayi PDF me nahi mila.' Apne se koi jawab mat banayein.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Streamlit Chat Interface UI
st.title("📚 Agentic RAG - PDF Chatbot")
st.write("Ye agent sirf aapki di gayi PDF ke mutabiq jawab dega.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("PDF ke baare me kuch bhi puchein..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Get Response from RAG Chain
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})