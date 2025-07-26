# run_loader.py

from Chatbot.loader import load_documents

if __name__ == "__main__":
    print("[INFO] Building FAISS vector store...")
    load_documents(embedding_model_name="all-MiniLM-L6-v2")
    print(" Vector store successfully created!")
