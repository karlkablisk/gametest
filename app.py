import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent
from dotenv import load_dotenv
import traceback

load_dotenv()

# Initialize the agent executor
agent_executor = agent.get_agent_executor()
user_input = []

# Streamlit Interface Configurations
with st.sidebar:
    debug_mode = st.checkbox("Debug Mode")  # To toggle debug mode on/off
    st_description = st.text_input("Enter description:")
    description = agent.CustomPromptTemplate.get_description(st_description)

# Main Code
try:
    if debug_mode:  # Only run this block if Debug Mode is checked
        raise Exception("Debug Exception")

    st.title("Langchain Agent")
    user_input = st.text_input("Enter your query:")

    # Initialize StreamlitCallbackHandler
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

    if st.button("Send"):
        with st.container():  # Wrap agent output in a container
            result = agent_executor.run(user_input, callbacks=[st_cb])
            st.write(result)

except Exception as e:
    if debug_mode:
        st.write("An error occurred:", str(e))
        st.write("Traceback:", traceback.format_exc())
