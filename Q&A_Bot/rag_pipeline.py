from langchain.chains import RetrievalQA
from Chatbot.vector_db import load_vector_store
from Chatbot.prompts import PROMPT_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0.3)

retriever = load_vector_store().as_retriever()
prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)

def get_rag_chain():
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )