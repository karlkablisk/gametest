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

load_dotenv()

# Initialize embeddings and LLM.
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

# Streamlit UI
st.title('AI Shadowrun')

uploaded_file = st.file_uploader("Choose a State of the Union file")
url = st.text_input("Paste a Ruff FAQ URL")
question = st.text_input("What's your question?")


if st.button('Run Query'):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    # Initialize tools list
    tools = []
    
    # Load State of the Union document
    if uploaded_file:
        text = uploaded_file.read().decode('utf-8')
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        class PageContent:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {}

        texts = splitter.split_documents([PageContent(text)])
        reference_file_db = FAISS.from_documents(texts, embeddings)  
        reference_file = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=reference_file_db.as_retriever())

        tools.append(
            Tool(
                name="File Question",
                func=reference_file.run,
                description="Question about the uploaded file."
            )
        )
        
    # Load Websiteloader FAQ
    if url:
        loader = WebBaseLoader(url)
        ruff_docs = loader.load()
        ruff_texts = text_splitter.split_documents(ruff_docs)
        ruff_db = FAISS.from_documents(ruff_texts, embeddings)  
        ruff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever())
        tools.append(
            Tool(
                name="URL Question",
                func=ruff.run,
                description="Question about the URL content."
            )
        )
        
    # Load Search Tool
    if url:
        search_docs = loader.load()
        search_texts = text_splitter.split_documents(ruff_docs)
        search_db = FAISS.from_documents(ruff_texts, embeddings)  
        search = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever())
        tools.extend(load_tools(["serpapi"]))

        
    # Initialize Agent
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    answer = agent.run(question)
    st.write(f'Answer: {answer}')