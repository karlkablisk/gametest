# app.py
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory  # Make sure to import this
import agent 
from agent import msgs, initialize_chain
import openai

from dotenv import load_dotenv
load_dotenv()

if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True)

if 'chat_memory' not in st.session_state:
    st.session_state['chat_memory'] = []  # Initialize chat_memory in session state

if 'conversation' not in st.session_state:
    st.session_state.conversation = agent.initialize_chain(st.session_state['memory'])

# Initialize the agent executor using the memory from session state
initialize_chain(st.session_state['memory'])

# Initialize the agent executor using the memory from session state
agent_executor = agent.initialize_chain(st.session_state['memory'])


# Initialize the agent executor
#agent_executor = agent.get_agent_executor()

user_input = []

#STREAMLIT INTERFACE
st.title("Langchain Agent")
#user_input = st.text_input("Enter your query:")

# Initialize StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)


user_input = st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?")
if user_input:

    try:
        with st.container():  
            result = agent_executor.run(user_input, callbacks=[st_cb]) #callbacks is what makes the thining code
            st.write(result)
            agent.memory.load_memory_variables([])
            st.session_state['chat_memory'] = agent.memory.chat_memory
    except openai.error.APIError as e:
        st.error(f"An error occurred: {e}")
        

with st.sidebar:
    st_description = st.text_input("Enter description:")
    description = agent.CustomPromptTemplate.get_description(st_description)
    print(agent.memory.buffer)

#st.write("Session State:", st.session_state)  # This will now include 'chat_memory'

# Debug Printout of ChatMessageHistory 
st.write("Loaded Tools:", tools_string())
#print(tools_string)
#st.write("Chat Message History:", msgs)


#st.write("Conversation Memory:", agent.memory.chat_memory)
