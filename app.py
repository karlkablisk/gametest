# app.py
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent 
from dotenv import load_dotenv

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
    with st.container():  # Wrap agent output in a container
        # Pass the StreamlitCallbackHandler in the callbacks argument callbacks=[st_cb] is what makes it work!
        result = agent_executor.run(user_input, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])


with st.sidebar:
    st_description = st.text_input("Enter description:")
    description = agent.CustomPromptTemplate.get_description(st_description)
    print(agent.memory.buffer)

st.write("Session State:", st.session_state)

# Debug Printout of ChatMessageHistory
st.write("Chat Message History:", msgs)

st.write("Conversation Memory:", agent.memory.chat_memory)