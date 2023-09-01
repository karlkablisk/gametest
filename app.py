import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent 
from agent import msgs
from dotenv import load_dotenv
import pymysql
import requests
import time


load_dotenv()

# MySQL Configuration
DB_HOST = 'mysql.kabliskkeep.com'
DB_USER = 'kabliskadmin'
DB_PASS = st.secrets["DB_PASS"]
DB_NAME = 'karlaidb'
TABLE_NAME = 'aidiscord_'

# Webhook Configuration
WEBHOOK_URL = st.secrets["WEBHOOK_URL"]


# Initialize the agent executor
agent_executor = agent.get_agent_executor()

def fetch_messages():
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
    cursor = conn.cursor()
    sql = f"SELECT username, message FROM {TABLE_NAME}"
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return rows

def send_to_discord(message):
    payload = {'content': message}
    requests.post(WEBHOOK_URL, json=payload)

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
    database_placeholder = st.empty()
    while True:
        database_msgs = fetch_messages()
        database_placeholder.write("Database Message History:", database_msgs)
        time.sleep(1)  # Refresh every 1 second


# Debug printouts
st.write("Session State:", st.session_state)
st.write("Chat Message History:", msgs)
st.write("Conversation Memory:", agent.memory.chat_memory)

# Output Database Messages
database_msgs = fetch_messages()
st.write("Database Message History:", database_msgs)
