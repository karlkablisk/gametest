#app.py
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import agent
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent executor
agent_executor = agent.get_agent_executor()

#STREAMLIT INTERFACE
st.title("Langchain Agent")

# Display a message
st.write("This tool let's you talk to the ai")

# Get user input
user_input = st.text_input("Enter your query:", key="user_input_key")

# Initialize StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

if st.button("Send"):
    with st.container():  # Wrap agent output in a container
        # Pass the StreamlitCallbackHandler in the callbacks argument callbacks=[st_cb] is what makes it work!
        result = agent_executor.run(user_input, callbacks=[st_cb])
        st.write(result)
        agent.memory.load_memory_variables([])

    # Clear the text input and focus back on it
    st.session_state.user_input_key = ""  # Clearing the textbox
    st.write('<input type="text" name="user_input_key" id="user_input_key" autofocus />', unsafe_allow_html=True)

with st.sidebar:
    st_description = st.text_input("Enter description:")
    description = agent.CustomPromptTemplate.get_description(st_description)
    print(agent.memory.buffer)
