import os
import openai
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain, PromptTemplate

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Class to handle the main chatbot functionalities
class MainChatbot:
    def __init__(self):
        self.conversation_history = []

        # Define prompt templates for the two chains
        self.health_info_prompt_template = PromptTemplate(
            template="Extract any health-related information from the following text. If there is no important health information, output 'empty'.\n\nText: {input}\nHealth Info:",
            input_variables=["input"]
        )
        self.answer_prompt_template = PromptTemplate(
            template="You are a friendly chatbot. Continue the conversation based on the following input: {input}\nBot:",
            input_variables=["input"]
        )

        # Initialize the LangChain models for each task
        self.health_info_chain = LLMChain(llm=OpenAI(temperature=0.7), prompt=self.health_info_prompt_template)
        self.answer_chain = LLMChain(llm=OpenAI(temperature=0.7), prompt=self.answer_prompt_template)

    def add_to_history(self, user_input, health_info, bot_response):
        self.conversation_history.append({
            'user_input': user_input,
            'health_info': health_info,
            'bot_response': bot_response
        })

    def generate_response(self, user_input):
        # Extract health-related information
        health_info = self.health_info_chain.run(input=user_input).strip()

        # Generate response based on user input
        bot_response = self.answer_chain.run(input=user_input).strip()

        self.add_to_history(user_input, health_info, bot_response)
        return {
            'user_input': user_input,
            'health_info': health_info if health_info else "No health information extracted.",
            'bot_response': bot_response
        }

    def get_conversation_history(self):
        return self.conversation_history

