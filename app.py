# app.py
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent 
from agent import msgs
from dotenv import load_dotenv

load_dotenv()

if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True)

# Initialize the agent executor using the memory from session state
agent_executor = agent.initialize_chain(st.session_state['memory'])


# Initialize the agent executor
agent_executor = agent.get_agent_executor()

user_input = []

#STREAMLIT INTERFACE
st.title("Langchain Agent")
user_input = st.text_input("Enter your query:")

# Initialize StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

if st.button("Send"):
    with st.container():  # Wrap agent output in a container
        # Pass the StreamlitCallbackHandler in the callbacks argument callbacks=[st_cb] is what makes it work!
        result = agent_executor.run(user_input, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])
        
        # Copying agent.memory.chat_memory to st.session_state
        st.session_state['chat_memory'] = agent.memory.chat_memory  # Add this line

with st.sidebar:
    st_description = st.text_input("Enter description:")
    description = agent.CustomPromptTemplate.get_description(st_description)
    print(agent.memory.buffer)

st.write("Session State:", st.session_state)  # This will now include 'chat_memory'

# Debug Printout of ChatMessageHistory 
st.write("Chat Message History:", msgs)

st.write("Conversation Memory:", agent.memory.chat_memory)
