#file imports
import toollist
from toollist import ALL_TOOLS, tools_string, tool_names

#Langchain imports
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.tools import tool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document, AgentAction, AgentFinish
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationChain

#other imports
import re
from typing import List, Union, Callable
from dotenv import load_dotenv
import random
from typing import Tuple, Optional
import streamlit as st

load_dotenv()


#ALL TOOLS

# Custom tools
@tool
def discover_item(user_context: str) -> str:
    """Determines what item is found based on user context.
    You decide what the name is, it will the become the {item_name}.
    You decide what the description of it, it will the become the {item_description}
    """
    item_name = f"{user_context}_item_{random.randint(1, 100)}"
    item_description = f"A special item found in {user_context}"
    return f"Item Name: {item_name}, Description: {item_description}"
    
@tool
def about_you(user_context: str) -> str:
    """Determines what item is found based on user context."""
    return f"Results"
    
tools = [] 

#tools_string
    
#TOOLS END

#LLM AND MODELS
main_model = "gpt-3.5-turbo"
strong_model = "gpt-4"
gpt35_16 = "gpt-3.5-turbo-16k"
gpt4_16 = "gpt-4-16k"
homemodel = "meta/llama-2" #need to edit this

llm = ChatOpenAI(model_name=main_model, temperature=0.2, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True,temperature=0.2)

#LLM END

#DATA STORAGE
msgs = ChatMessageHistory(key="memory_key")
embeddings = OpenAIEmbeddings()

#memory start
memory = ChatMessageHistory(memory_key="history", chat_memory=msgs, input_key="input", return_messages=True)


docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS)]
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever()

  #STORE TOOL DATA IN VECTORSTORE
def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [tools[d.metadata["index"]] for d in docs if 0 <= d.metadata["index"] < len(tools)]




#DATA STORAGE END


# Custom Prompt Template
template = """
Your name is {name}.
You are a helpful AI assistant, but you never let on you are AI, for all intents and purposes you behave and act like Anna.
Personality: You are a shy girl, talking to this person for the first time. You want to introduce yourself and get to know them better.

Your description is as follows: {description}
Your traits are {traits}
Your likes are {likes}

You have access to the following tools:

{tools}

Use the following format when talking:
For simple statements you can just reply back. Skip to final answer and give a solid reply.

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, only if it sounds like its related to a tool, should you use one of [{tool_names}]
If you try to access a tool that doesnt exsist, its ok, it means there isn't a tool for the situation and you should just reply back normally.
only do the discover_item action once, then make up a name, a description and go straight to the final answer.

Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools_getter: Callable
    #template variables go here
    name = "Anna"
    traits = "patient, knowledgeable, encouraging"
    likes = "going on walks to the beach, tea, comic books"
    #complex variables that can be filled in by the user go here as a function but are otherwise the same
    def get_description(self, st_description=None):
        return st_description or "A creative, knowledgeable, and encouraging high school teacher."

    def format(self, **kwargs) -> str:
        #chat_history = memory.get('history')  # Fetch the chat history
        #kwargs["history"] = "\n".join(chat_history)
        kwargs["name"] = self.name
        kwargs["traits"] = self.traits
        kwargs["likes"] = self.likes
        kwargs["description"] = self.get_description(kwargs.get("st_description", None))
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    input_variables=["input", "intermediate_steps"]
)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)

#TEMPLATE END


#LLM CHAIN
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)


# Output Parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Use a more robust regex pattern to match action and action input
        regex = r"Action\s*:\s*(.*?)\nAction\s*Input\s*:\s*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        
        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')
        
        return AgentAction(
            tool=action, tool_input=action_input, log=llm_output
        )



output_parser = CustomOutputParser()

# OUTPUT PARSER END



#AGENT AND EXECUTOR

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
    handle_parsing_errors=True,
    verbose=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

def get_agent_executor():
    return agent_executor
    
#AGENT END





