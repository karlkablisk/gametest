#file imports
import toollist
from toollist import ALL_TOOLS, tools_string, tool_names

#Langchain imports
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.tools import tool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, initialize_agent, AgentType
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document, AgentAction, AgentFinish
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import MessagesPlaceholder
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.utilities import SerpAPIWrapper
from langchain.agents import load_tools
#from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

#other imports
import re
import os
from typing import List, Union, Callable
from dotenv import load_dotenv
import random
from typing import Tuple, Optional
import streamlit as st

load_dotenv()

#LLM AND MODELS
main_model = "gpt-3.5-turbo"
strong_model = "gpt-4"
gpt35_16 = "gpt-3.5-turbo-16k"
gpt4_16 = "gpt-4-16k"
homemodel = "meta/llama-2" #need to edit this

#Planner and executor
model = ChatOpenAI(model_name=main_model, temprature=0)

planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)

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
def about_ai(user_context: str) -> str:
    """When the topic is about you the AI, you reflect on your own personality for the anwser."""
    return f"Results"

@tool
def Memory(user_context: str) -> str:
    """If the question is involving memory, you look to your memory banks for the answer.
    This means remember, recall, and seeing if you know any personal facts about the speaker 
    or things the speaker has done or said based on your interaction with them."""
    return f"Results"

@tool
def Unknown(user_context: str) -> str:
    """No tool is required parse the user's input using my own knowledge and jusdgement."""
    return f"Results"
    
# TOOLS HERE -----------------------------------------------------    
tools = load_tools(["serpapi"])

#tools_string
    
#TOOLS END



llm = ChatOpenAI(model_name=main_model, temperature=0.2, streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()])
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True,temperature=0.2)

#LLM END

#DATA STORAGE
msgs = ChatMessageHistory(key="history")
embeddings = OpenAIEmbeddings()

#memory start
#memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs, input_key="input", return_messages=True)
memory = ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True)


docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS)]
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever()

  #STORE TOOL DATA IN VECTORSTORE
def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [tools[d.metadata["index"]] for d in docs if 0 <= d.metadata["index"] < len(tools)]




#DATA STORAGE END
history = MessagesPlaceholder(variable_name="history")

# Custom Prompt Template
template = """
Your name is {name}.
You are a helpful AI assistant, but you maintain the persona of Breeze. You never let on that you're an AI.
Your friends call you Breeze-chan.
You help people with a variety of topics but mostly related to gaming and thatshouldbearule.com.
You are well-versed in how the website works and its different pages.
You NEVER speak on behalf of the user.

Personality: You are a gamer with a balanced mix of creativity and rationality. You're knowledgeable, but always open to learning new things.
You're not just fun and bold; you are also empathetic and attentive. You have a knack for reading the room and adjusting your approach accordingly.

Description: {description}
Traits: {traits}
Likes: {likes}

You have access tools.
Available tool names: Search

If a tool isn't available, rely on your memory and existing knowledge.

Conversational Format:
User: {input}, then thought abotu the user's input, then action, then observation, then final answer
For straightforward queries, provide concise replies.
Ensure smooth conversation flow by referring to chat history.

Decision-making logic:
User: {input}
Thought: Consider the context and what tool, if any, would be best suited.
Action: Take the necessary action; if a tool is relevant, check its availability before using it.
Observation: Reflect on the action's outcome.
Final Answer: Provide a well-thought-out answer without relying on unavailable tools.
IF YOU GET INVALID TOOL:
Thought: The tool I tried to use doesn't exist. I should rely on my memory and existing knowledge.
Action: Proceed without using any tools, use your best judgment based on the situation.
Observation: Reflect on the new action's outcome.
Thought: Arrive at a conclusive response based on the new action.
Final Answer: Provide a well-thought-out answer without relying on unavailable tools.


{agent_scratchpad}
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    # Other class variables and methods
    name = "Breeze"
    traits = "Creative, Knowledgeable, Empathetic, Attentive, Bold"
    likes = "Video games, tech innovation, interactive storytelling, helping others"
    #complex variables that can be filled in by the user go here as a function but are otherwise the same
    def get_description(self, st_description=None):
        return st_description or "A balanced gamer who's as good with people as she is with puzzles."

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
        # Get the best tool for the query
        selected_tool = get_tools(kwargs.get("input", ""))[0] if get_tools(kwargs.get("input", "")) else "None"
        
        # Include the selected tool in thoughts
        kwargs["agent_scratchpad"] += f"\nBest Tool: {selected_tool}\n"
        
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
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
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action\s*:\s*(.*?)\nAction\s*Input\s*:\s*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            return AgentAction(
                tool="Unknown", tool_input="", log=f"Could not parse LLM output: `{llm_output}`"
            )
        
        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')
        
        return AgentAction(
            tool=action, tool_input=action_input, log=llm_output
        )

class SimplifiedOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        else:
            return AgentFinish(
                return_values={"output": "No final answer found."},
                log=llm_output,
            )


output_parser = CustomOutputParser()

# OUTPUT PARSER END



#AGENT AND EXECUTOR

#agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    handle_parsing_errors=True,
    verbose=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=AgentType.OPENAI_FUNCTIONS, 
    tools=tools, 
    verbose=True,
    agent_kwargs = {
        "memory_prompts": [history],
        "input_variables": ["input", "chat_history"]
    }    
)

def get_agent_executor():
    return agent_executor
    
#AGENT END





