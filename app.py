import streamlit as st
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.document_loaders import TextLoader, WebBaseLoader
from dotenv import load_dotenv
from langchain.utilities import SerpAPIWrapper
from langchain.agents import load_tools
import numpy as np


import requests
import json

load_dotenv()

# Initialize embeddings and LLM.
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

# Streamlit UI
st.title('Arrowtokyo AI Chat')

# Fetching data directly from the URL and storing it as 'text'
response = requests.get('https://arrowtokyo.com/en/arrow/rest/products')
text = json.dumps(response.json())

question = st.text_input("What's your question?")

embeddings_file_data = None

if st.button('Run Query'):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    # Initialize tools list
    tools = []
    
    # Use the text fetched from the URL
    if text:
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        class PageContent:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {}

        texts = splitter.split_documents([PageContent(text)])
        reference_file_db = FAISS.from_documents(texts, embeddings)  
        
        # Convert the embeddings to a .npy file and allow the user to download it
        np.save('embeddings.npy', reference_file_db.embeddings)
        st.download_button(
            label="Download embeddings file",
            data=bytes(open('embeddings.npy', 'rb').read()),
            file_name='embeddings.npy',
            mime='application/octet-stream'
        )
        
         # Store the embeddings data in a variable
        embeddings_file_data = bytes(np.save_to_buffer(reference_file_db.embeddings))
        
        reference_file = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=reference_file_db.as_retriever())
    

        tools.append(
            Tool(
                name="File Question",
                func=reference_file.run,
                description="Question about the uploaded file."
            )
        )
        
    # Initialize Agent
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    answer = agent.run(question)
    st.write(f'Answer: {answer}')

if embeddings_file_data and st.button('Download Embeddings File'):
    st.download_button(
        label="Download embeddings file",
        data=embeddings_file_data,
        file_name='embeddings.npy',
        mime='application/octet-stream'
    )