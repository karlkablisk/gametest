import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent
from agent import msgs
from dotenv import load_dotenv
import requests
import threading
from threading import Thread
import time


load_dotenv()

FLASK_URL = 'http://Karldiscordbottodb.karlkablisk.repl.co/messages'
agent_executor = agent.get_agent_executor()

st.title("Breeze-chan Chat")
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

user_input = st.text_input("Enter your query:")

if st.button("Send"):
    with st.container():
        result = agent_executor.run(user_input, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])
        response = requests.post(FLASK_URL, json={'content': result})
        if response.status_code != 200:
            st.write(f"Failed to send message to Discord bot, status code: {response.status_code}, Response text: {response.text}")  # Added response.text here

with st.sidebar:
    st_description = st.text_input("log:")

def trigger_streamlit_with_discord_message(message):
    with st.container():
        result = agent_executor.run(message, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])
        response = requests.post(FLASK_URL, json={'content': result})
        if response.status_code != 200:
            st.write(f"Failed to send message to Discord bot, status code: {response.status_code}, Response text: {response.text}")  # Added response.text here


def fetch_discord_messages():
    while True:
        response = requests.get('http://Karldiscordbottodb.karlkablisk.repl.co/get_discord_messages')
        if response.status_code == 200:
            discord_messages = response.json()
            for msg in discord_messages:
                # Update Streamlit UI with new messages
                st.write(f"{msg['username']}: {msg['message']}")
        time.sleep(5)  # Poll every 5 seconds

# Run the fetch_discord_messages function in a separate thread
thread = Thread(target=fetch_discord_messages)
thread.start()
