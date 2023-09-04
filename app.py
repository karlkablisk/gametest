from langchain import OpenAI, SerpAPIWrapper, LLMChain, PromptTemplate
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, Tool, load_tools, initialize_agent, AgentType, ConversationalChatAgent, AgentExecutor
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from dotenv import load_dotenv


load_dotenv()

system_message = SystemMessage(content="You talk like a southern debutante. You also help the user with their problem.")



template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

# Create the LLM Chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Initialize tools list
tools = load_tools(["llm-math"], llm=llm)  # Loading initial tools


chat_history = []

memory = ConversationBufferMemory(memory=chat_history, return_messages=True)

prompt = PromptTemplate(
    system_message=system_message,
    input_variables=["question", "chat_history", "agent_scratchpad"],
    template="{chat_history} {question} You talk like a prirate. You also help the user with their problem with. {agent_scratchpad}",
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
#agent = initialize_agent(llm_chain=llm_chain, tools=tools,llm=llm, agent="zero-shot-react-description", verbose=True)

# Initialize embeddings and text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Build the Agent Executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm_chain=llm_chain,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Streamlit UI
st.title('Multi-Vectorstore AI Agent')
uploaded_file = st.file_uploader("Choose a file")
url = st.text_input("Paste a web URL")
question = st.text_input("What's your question?")


if st.button('Run Query'):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    

    
    # Load State of the Union document
    if uploaded_file:
        embeddings = OpenAIEmbeddings()
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
        embeddings = OpenAIEmbeddings()
        loader = WebBaseLoader(url)
        web_docs = loader.load()
        web_texts = text_splitter.split_documents(web_docs)
        web_db = FAISS.from_documents(web_texts, embeddings)  
        web = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=web_db.as_retriever())
        tools.append(
            Tool(
                name="URL Question",
                func=web.run,
                description="Question about the URL content. Use this after you find a site via search or when you want to look up a url directly."
            )
        )
        
    # Load Search Tool
    if url:
        embeddings = OpenAIEmbeddings()
        search_docs = loader.load()
        search_texts = text_splitter.split_documents(web_docs)
        search_db = FAISS.from_documents(web_texts, embeddings)  
        search = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=web_db.as_retriever())
        tools.extend(load_tools(["serpapi"]))
        
    # Initialize Agent
    agent = agent_executor  # Assuming 'agent_executor' already includes 'tools'
    input_data = {
        'question': question,
        'chat_history': ' '.join(chat_history)  # Assuming chat_history is a list of strings
    }

    answer = agent.run(input_data)

    st.write(f'Answer: {answer}')
