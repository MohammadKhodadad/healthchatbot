import os
import openai
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain, PromptTemplate
# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


from modules.rag import RAG
rag=RAG()
print(rag.ask_question("My blood pressure was low lately"))