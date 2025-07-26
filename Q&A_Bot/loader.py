import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_documents(embedding_model_name="all-MiniLM-L6-v2"):
    documents_path = "Chatbot/documents"
    if not os.path.exists(documents_path):
        raise ValueError(f"{documents_path} does not exist. Create and add .txt or .pdf files.")

    print("[INFO] Loading documents...")
    loader = DirectoryLoader(documents_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    print(f"[INFO] Loaded {len(documents)} documents. Splitting and embedding...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("Chatbot/faiss_index")
    print("[INFO] FAISS index saved to Chatbot/faiss_index")
