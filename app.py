import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent
from agent import msgs
from dotenv import load_dotenv
import requests
import time

load_dotenv()

# Webhook Configuration
WEBHOOK_URL = st.secrets["WEBHOOK_URL"]

# URL for the Flask app
FLASK_URL = 'http://Karldiscordbottodb.karlkablisk.repl.co/messages'  # Replace with your Flask app URL

# Initialize the agent executor
agent_executor = agent.get_agent_executor()

# Streamlit UI
st.title("Langchain Agent")
user_input = st.text_input("Enter your query:")

# Initialize StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

if st.button("Send"):
    with st.container():
        result = agent_executor.run(user_input, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])
        send_to_discord(result)  # Sending to Discord
        # Copy agent.memory.chat_memory to st.session_state
        st.session_state['chat_memory'] = agent.memory.chat_memory

# Sidebar
with st.sidebar:
    st_description = st.text_input("log:")
    database_msgs = fetch_messages()
    st.write("Message History:", database_msgs)  # This replaces `database_placeholder.write`

    # New debug button
    if st.button("Debug Test"):
        payload = {'content': 'Debug message from Streamlit'}
        response = requests.post(FLASK_URL, json=payload)
        st.write(f"Debug Test Response: {response.status_code}")


# Debug printouts
st.write("Session State:", st.session_state)
st.write("Chat Message History:", msgs)
st.write("Conversation Memory:", agent.memory.chat_memory)

# Output Messages
database_msgs = fetch_messages()
st.write("Message History:", database_msgs)


def fetch_messages():
    response = requests.get(FLASK_URL)
    if response.status_code == 200:
        return response.json()
    else:
        return []

def send_to_discord(message):
    payload = {'content': message}
    requests.post(WEBHOOK_URL, json=payload)

def periodic_fetch():
    while True:
        new_msgs = fetch_messages()
        if new_msgs:  # Check if there are new messages
            for msg in new_msgs:
                user_input = msg['message']
                result = agent_executor.run(user_input, callbacks=[st_cb])
                send_to_discord(result)
            time.sleep(2)  # Sleep for 2 seconds before fetching again
        else:
            time.sleep(2)

if __name__ == '__main__':
    periodic_fetch()