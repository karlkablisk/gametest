# app.py
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent 
from agent import msgs
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent executor
agent_executor = agent.get_agent_executor()

user_input = []

# Initialize session state if not existent
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = []

#STREAMLIT INTERFACE
st.title("Langchain Agent")
user_input = st.text_input("Enter your query:")

# Initialize StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

if st.button("Send"):
    with st.container():  
        result = agent_executor.run(user_input, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])
        
        # Copy the chat memory to session_state and msgs
        st.session_state.chat_memory = agent.memory.chat_memory
        if agent.memory.chat_memory:  # Check if chat_memory is not empty
            if agent.memory.chat_memory.messages:  # Check if messages are not empty
                last_message = agent.memory.chat_memory.messages[-1]
                msgs.add_message(last_message)


with st.sidebar:
    st_description = st.text_input("Enter description:")
    description = agent.CustomPromptTemplate.get_description(st_description)
    print(agent.memory.buffer)

# Debug printouts
st.write("Session State:", st.session_state)
st.write("Chat Message History:", msgs)
st.write("Conversation Memory:", agent.memory.chat_memory)
