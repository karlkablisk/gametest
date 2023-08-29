import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent executor
agent_executor = agent.get_agent_executor()

# Initialize StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

#STREAMLIT INTERFACE
st.title("Langchain Agent")
st.write("This tool let's you talk to the ai")

input_placeholder = st.empty()
user_input = input_placeholder.text_input("Enter your query:", key="user_input_key")

if st.button("Send"):
    with st.container():  # Wrap agent output in a container
        result = agent_executor.run(user_input, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])

    # Clear the text input
    input_placeholder.text_input("Enter your query:", value='', key="user_input_key")

with st.sidebar:
    st_description = st.text_input("Enter description:")
    description = agent.CustomPromptTemplate.get_description(st_description)
    print(agent.memory.buffer)
