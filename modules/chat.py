import os
import openai
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain, PromptTemplate
from .rag import RAG
# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Class to handle the main chatbot functionalities
class MainChatbot:
    def __init__(self):
        self.conversation_history = []

        # Define prompt templates for the two chains
        self.health_info_prompt_template = PromptTemplate(
            template="You are an information extractor. There is this text and it might have something that is related to the person health.If there is something clearly stated in the text, extract and return it. If there is no important health information, empty text like '' .\n\nText: {input}\nHealth Info:",
            input_variables=["input"]
        )
        self.answer_prompt_template = PromptTemplate(
            template="You are a friendly chatbot, you also have a medical side chatbot that guides you sometimes. Continue the conversation based on the following input: {input}\n medical advice: {medical_advice}\nBot:",
            input_variables=["input",'medical_advice']
        )

        # Initialize the LangChain models for each task
        self.health_info_chain = LLMChain(llm=OpenAI(temperature=0.7), prompt=self.health_info_prompt_template)
        self.answer_chain = LLMChain(llm=OpenAI(temperature=0.7), prompt=self.answer_prompt_template)
        print("everything ready before rag")
        self.rag = RAG()
        print('rag ready!')
    def add_to_history(self, user_input, health_info,medical_advice, bot_response):
        self.conversation_history.append({
            'user_input': user_input,
            'health_info': health_info,
            'medical_advice':medical_advice,
            'bot_response': bot_response
        })

    def generate_response(self, user_input):
        # Extract health-related information
        health_info = self.health_info_chain.run(input=user_input).strip()
        if len(health_info)>0:
            medical_advice = self.rag.ask_question(health_info)
        else:
            medical_advice = ''
        inputs = {
            "input": user_input,
            "medical_advice": medical_advice
        }
        bot_response = self.answer_chain.run(input=user_input,medical_advice=medical_advice).strip()

        self.add_to_history(user_input, health_info,medical_advice, bot_response)
        return {
            'user_input': user_input,
            'health_info': health_info if health_info else "No health information extracted.",
            'medical_advice':medical_advice,
            'bot_response': bot_response,
            
        }

    def get_conversation_history(self):
        return self.conversation_history

