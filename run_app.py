import  streamlit as st
from modules.chat import *

# Streamlit app
def main():
    st.title("Healthcare Chatbot")

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MainChatbot()
    
    user_input = st.text_input("You: ", "")
    if st.button("Send"):
        if user_input:
            response = st.session_state.chatbot.generate_response(user_input)
            st.write(f"Chatbot: {response}")
    
    if st.button("Show Conversation History"):
        conversation_history = st.session_state.chatbot.get_conversation_history()
        for user_msg, bot_msg in conversation_history:
            st.write(f"You: {user_msg}")
            st.write(f"Chatbot: {bot_msg}")

if __name__ == "__main__":
    main()
