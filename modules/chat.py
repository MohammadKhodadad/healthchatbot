import os
import openai
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain, PromptTemplate
# from langchain_openai import ChatOpenAI
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
            template="You are an information extractor. There is this part of a conversation where a person says something and it might have something that is related to the person health. If there is something clearly stated in the text, extract and return it. If there is no important health information, return '' .\n\nText: {input}\nHealth Info:",
            input_variables=["input"]
        )
        self.answer_prompt_template = PromptTemplate(
            template="You are a friendly chatbot, you also have a medical side chatbot that guides you sometimes. but you are the only one communicating to the person Continue the conversation based on the input, medical advice from the sidechat bot and history of your chat. You just have to give them a response based on the current query.\ninput: {input}\n medical advice: {medical_advice}\nConversation history: {conversation_history}\nBot:",
            input_variables=["input",'medical_advice','conversation_history']
        )

        # Initialize the LangChain models for each task
        # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=4000).bind(
        #     response_format={"type": "json_object"}
        # )
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
        if len(health_info)>4:
            medical_advice = self.rag.ask_question(health_info)
        else:
            medical_advice = ''
        bot_response = self.answer_chain.run(input=user_input,medical_advice=medical_advice,conversation_history=self.conversation_history).strip()

        self.add_to_history(user_input, health_info,medical_advice, bot_response)
        return {
            'user_input': user_input,
            'health_info': health_info,
            'medical_advice':medical_advice,
            'bot_response': bot_response,
            
        }

    def get_conversation_history(self):
        return self.conversation_history

