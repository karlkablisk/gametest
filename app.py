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
from langchain.callbacks import StreamlitCallbackHandler
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputParserException

import numpy as np
from io import BytesIO

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

class AutoFixingOutputParser:
    def __init__(self, original_parser, fallback_llm):
        self.original_parser = original_parser
        self.fallback_llm = fallback_llm

    def parse(self, output):
        try:
            return self.original_parser.parse(output)
        except OutputParserException as e:
            # Here, you might invoke the fallback_llm to attempt to fix the output, 
            # or implement other error handling logic
            # For the sake of the example, I'm simply re-raising the exception
            raise e

# Initialize your existing output parser and the fallback LLM
original_parser = PydanticOutputParser(pydantic_object=ExpectedOutputSchema)
fallback_llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

# Create the auto-fixing output parser
auto_fixing_parser = AutoFixingOutputParser(original_parser, fallback_llm)

# Initialize StreamlitCallbackHandler
st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

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
        
        # Store the embeddings data in a variable for later use
        embeddings_file_data = reference_file_db.embeddings
        
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
    st.write(f'Answer: {answer}', callbacks=[st_cb])

if embeddings_file_data is not None and st.button('Download Embeddings File'):
    # Convert the embeddings to a .npy file and allow the user to download it
    buffer = BytesIO()
    np.save(buffer, embeddings_file_data)
    buffer.seek(0)
    st.download_button(
        label="Download embeddings file",
        data=buffer.read(),
        file_name='embeddings.npy',
        mime='application/octet-stream'
    )


