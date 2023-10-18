import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
import agent
from agent import MyCustomCallback
from dotenv import load_dotenv
import requests
from threading import Thread
import time

load_dotenv()

# Instantiate MyCustomCallback
my_custom_callback_instance = MyCustomCallback()

# URLs configuration
FLASK_URL = 'http://Karldiscordbottodb.karlkablisk.repl.co/messages'
DISCORD_FETCH_URL = 'http://Karldiscordbottodb.karlkablisk.repl.co/get_discord_messages'
WEBHOOK_URL = st.secrets["WEBHOOK_URL"]  # Load the Webhook URL from secrets

# Initialize the agent executor
agent_executor = agent.get_agent_executor()

# Initialize Streamlit UI elements
st.title("Breeze-chan Chat")
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
user_input = st.text_input("Enter your query:")

# Function to send the AI response to Discord via Webhook
def send_to_discord(message):
    payload = {'content': message}
    requests.post(WEBHOOK_URL, json=payload)

# If the "Send" button is clicked
if st.button("Send"):
    with st.container():
        result = agent_executor.run(user_input, callbacks=[])
        my_custom_callback_instance.set_st_cb_result(result)
        
        # Run the agent again with both callbacks
        #agent_executor.run(user_input, callbacks=[st_cb, my_custom_callback_instance])
        
        agent.memory.load_memory_variables([])
        send_to_discord(result)

        #get output
        ai_output = result["output"]
        st.write(f"She said: {ai_output}")


# Sidebar configuration
with st.sidebar:
    st_description = st.text_input("log:")

# Function to fetch Discord messages and display them in Streamlit
def fetch_discord_messages():
    while True:
        response = requests.get(DISCORD_FETCH_URL)
        if response.status_code == 200:
            discord_messages = response.json()
            for msg in discord_messages:
                st.write(f"{msg['username']}: {msg['message']}")
        time.sleep(5)  # Poll every 5 seconds

# Start a separate thread to run the fetch_discord_messages function
thread = Thread(target=fetch_discord_messages)
thread.start()
