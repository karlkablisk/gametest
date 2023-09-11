import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent
from agent import msgs
from dotenv import load_dotenv
import requests
import subprocess

load_dotenv()

# Webhook Configuration
WEBHOOK_URL = st.secrets["WEBHOOK_URL"]

# URL for the Flask app
FLASK_URL = 'http://Karldiscordbottodb.karlkablisk.repl.co/messages'  # Replace with your Flask app URL

# Initialize the agent executor
agent_executor = agent.openai_agent()

def fetch_messages():
    response = requests.get(FLASK_URL)
    if response.status_code == 200:
        return response.json()
    else:
        return []

def send_to_discord(message):
    payload = {'content': message}
    requests.post(WEBHOOK_URL, json=payload)

# Streamlit UI
st.title("Breeze-chan Chat")
user_input = st.text_input("Enter your query:")

# Initialize StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

if st.button("Send"):
    with st.container():
        result = agent_executor.run(user_input, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])
        send_to_discord(result)  # Sending to Discord
        st.session_state['chat_memory'] = agent.memory.chat_memory

# Sidebar
with st.sidebar:
    st_description = st.text_input("log:")
    database_msgs = fetch_messages()
    #st.write("Message History:", database_msgs)

    if st.button("Debug Test"):
        payload = {'content': 'Debug message from Streamlit'}
        response = requests.post(FLASK_URL, json=payload)
        st.write(f"Debug Test Response: {response.status_code}")
        try:
            response_data = response.json()
            received_message = response_data.get('message', 'No message received')
            st.write("Received message:", received_message)
        except ValueError:
            st.write("Invalid JSON response from server")

# Debug printouts
#st.write("Session State:", st.session_state)
#st.write("Chat Message History:", msgs)
#st.write("Conversation Memory:", agent.memory.chat_memory)

# Output Messages
database_msgs = fetch_messages()
#st.write("Message History:", database_msgs)

# Trigger Streamlit with Discord message
def trigger_streamlit_with_discord_message(message):
    with st.container():
        result = agent_executor.run(message, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])
        send_to_discord(result)
        st.session_state['chat_memory'] = agent.memory.chat_memory

# Test trigger from Discord
#if st.button("Test Streamlit Trigger from Discord"):
#    test_message = "Test message from Discord"
   # trigger_streamlit_with_discord_message(test_message)
