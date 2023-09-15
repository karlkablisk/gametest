import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent
from agent import msgs
from dotenv import load_dotenv
import requests
import subprocess
import time

def fetch_discord_messages():
    while True:
        response = requests.get('http://Karldiscordbottodb.karlkablisk.repl.co/get_discord_messages')
        if response.status_code == 200:
            discord_messages = response.json()
            for msg in discord_messages:
                trigger_streamlit_with_discord_message(msg['message'])
        time.sleep(5)  # Poll every 5 seconds

# Start a thread to fetch Discord messages
from threading import Thread
thread = Thread(target=fetch_discord_messages)
thread.start()

load_dotenv()

WEBHOOK_URL = st.secrets["WEBHOOK_URL"]
FLASK_URL = 'http://Karldiscordbottodb.karlkablisk.repl.co/messages'

agent_executor = agent.get_agent_executor()

def fetch_messages():
    response = requests.get(FLASK_URL)
    if response.status_code == 200:
        return response.json()
    else:
        return []
    
def fetch_discord_messages():
    while True:
        response = requests.get('http://Karldiscordbottodb.karlkablisk.repl.co/get_messages')
        if response.status_code == 200:
            discord_messages = response.json()
            for msg in discord_messages:
                trigger_streamlit_with_discord_message(msg)
        time.sleep(5)  # Poll every 5 seconds


def send_to_discord(message):
    payload = {'content': message}
    requests.post(WEBHOOK_URL, json=payload)

st.title("Breeze-chan Chat")

st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

user_input = st.text_input("Enter your query:")

if st.button("Send"):
    with st.container():
        result = agent_executor.run(user_input, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])
        send_to_discord(result)
        st.session_state['chat_memory'] = agent.memory.chat_memory

with st.sidebar:
    st_description = st.text_input("log:")
    database_msgs = fetch_messages()

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

def trigger_streamlit_with_discord_message(message):
    with st.container():
        result = agent_executor.run(message, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])
        send_to_discord(result)
        st.session_state['chat_memory'] = agent.memory.chat_memory
