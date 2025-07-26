import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DB_PATH = os.path.join("Chatbot", "vector_db")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(docs):
    vector_store = FAISS.from_documents(docs, EMBEDDING_MODEL)
    vector_store.save_local(VECTOR_DB_PATH)

def load_vector_store():
    return FAISS.load_local(
        VECTOR_DB_PATH,
        EMBEDDING_MODEL,
        allow_dangerous_deserialization=True 
    )
