# app.py
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent 
from agent import msgs
from dotenv import load_dotenv
import copy

load_dotenv()


# Initialize the agent executor
agent_executor = agent.get_agent_executor()

user_input = []



#STREAMLIT INTERFACE
st.title("Langchain Agent")
user_input = st.text_input("Enter your query:")

# Initialize StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

if st.button("Send"):
  with st.container(): 
    result = agent_executor.run(user_input, callbacks=[st_cb])
    
    # Update session state and history
    update_session_and_history(agent.memory)  
    
    st.write(result)

# Function to update session and history  
def update_session_and_history(memory):
  st.session_state['conversation_memory'] = copy.deepcopy(memory.chat_memory)
  msgs.append(copy.deepcopy(memory.chat_memory))
        


with st.sidebar:
    st_description = st.text_input("Enter description:")
    description = agent.CustomPromptTemplate.get_description(st_description)
    print(agent.memory.buffer)


st.write("Session State:", st.session_state)

# Debug Printout of ChatMessageHistory 
st.write("Chat Message History:", msgs)

st.write("Conversation Memory:", agent.memory.chat_memory)
