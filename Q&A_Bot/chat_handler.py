import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Get API keys
openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

llm = None
llm_name = "Not initialized"

if google_key:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_key)
    llm_name = "Google Gemini Pro"
if openai_key:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_key)
    llm_name = "OpenAI GPT-3.5 Turbo"
else:
    raise ValueError("Neither OpenAI nor Google API key found in .env file!")

# --- Load FAISS Vector Store ---
def load_vector_store():
    if not os.path.exists("Chatbot/faiss_index"):
        raise FileNotFoundError("Vector store missing. Run run_loader.py to create FAISS index.")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("Chatbot/faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db

retriever = load_vector_store().as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Handle User Query ---
def handle_chat_query(user_query):
    return qa_chain.run(user_query)

# --- Return Active LLM Name ---
def get_llm_model_name():
    return llm_name
