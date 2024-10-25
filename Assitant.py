# Import environmet variables
import os
from dotenv import load_dotenv

# Manually reset the environment variable
if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]
# Load environment variables from the .env file
load_dotenv()  # This loads the .env file in the current directory

# Verify that the environment variable is loaded
api_key = os.getenv("ANTHROPIC_API_KEY")

if api_key:
    print("API key loaded successfully.")
else:
    print("API key not found. Make sure the .env file is properly configured.")


# Set up Public model
from langchain_anthropic import ChatAnthropic
claude_model = ChatAnthropic(model_name="claude-3-sonnet-20240229")

# Start the Ollama server
import subprocess

def start_ollama_server():
    try:
        # Run the "ollama serve" command
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Ollama server started successfully.")
        return process
    except Exception as e:
        print(f"Error starting Ollama server: {e}")

# Start the server
ollama_process = start_ollama_server()

def is_ollama_running():
    try:
        # Run the "pgrep" command to check for the "ollama" process
        result = subprocess.run(["pgrep", "-f", "ollama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("Ollama server is running.")
            return True
        else:
            print("Ollama server is not running.")
            return False
    except Exception as e:
        print(f"Error checking if Ollama is running: {e}")
        return False
is_ollama_running()

# Set up Local LLM
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

llama_model = ChatOllama(model="llama3.2")
response = llama_model.invoke([HumanMessage(content="Are you there?")])
print(response.content)

# Set up Gmail tools
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_google_community import GmailToolkit
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

# Set up google calendar tools
from composio_langchain import ComposioToolSet, Action
tool_set = ComposioToolSet()
google_calendar_tools = tool_set.get_tools(actions=[
    Action.GOOGLECALENDAR_CREATE_EVENT,
    Action.GOOGLECALENDAR_DELETE_EVENT,
    Action.GOOGLECALENDAR_FIND_FREE_SLOTS,
    Action.GOOGLECALENDAR_UPDATE_EVENT
])

# Setup Internet search tool
from langchain_community.tools import DuckDuckGoSearchResults

internet_search_tool = DuckDuckGoSearchResults()

#Set up PDF tools
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load PDFs and create the vector store
pdf_loader = PyPDFLoader("../Lectures/Lecture 1.pdf")
documents = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)
vector_store = FAISS.from_documents(split_docs, embedding_model)

# configure retreival QA
pdf_qa_chain = RetrievalQA.from_chain_type(
    llm=llama_model,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)
# Add the PDF QA tool to the agent
class PDFQATool:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def run(self, question):
        return self.qa_chain.run(question)


pdf_qa_tool = PDFQATool(pdf_qa_chain)

# Combine all tools
tools = toolkit.get_tools()
tools.append(internet_search_tool)
tools.extend(google_calendar_tools)
# tools.append(pdf_qa_tool)

# Set up agent with claude model

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

claude_memory = MemorySaver() # Initialize memory saver

claude_agent = create_react_agent(claude_model, tools, checkpointer=claude_memory)

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
class Output(BaseModel):
    tools: str = Field(description="Tools used")
    AI_message: str = Field(description="AI message")

parser = JsonOutputParser(pydantic_object=Output)

from langchain.agents import create_react_agent
chain = claude_model
chained_agent = create_react_agent(chain, tools, output_parser=parser)

import json
config = {"configurable": {"thread_id": "xyz456"}}
temp = "hello "
for chunk in claude_agent.stream(
{"messages": [HumanMessage(content=temp)]}, config
):
    data = chunk
    print(type(data))
    print(data)
    print("----")

# Set up agent with llama model
from langchain_core.messages import SystemMessage

llama_memory = MemorySaver() # Initialize memory saver
system_message = SystemMessage(content="You are a helpful assistant.")

llama_agent = create_react_agent(llama_model, tools, checkpointer=llama_memory)#, state_modifier=SystemMessage)

# Check private data with Chain of Thought
def check_private_data_with_cot(message: str, local_llm: Callable) -> bool:
    """Check if the message contains private data using the local LLM with Chain of Thought prompting."""
    private_data_prompt = (
        "Analyze the following message to determine if it contains private or sensitive information. "
        "Private information includes personal identifiers (e.g., social security numbers, passport numbers), "
        "contact information (e.g., phone numbers, email addresses), financial details (e.g., credit card numbers, bank account information), "
        "or personal health information. Here’s how you should think step by step:\n\n"
        "1. Break down the message and identify any specific pieces of information that might fall into one of these categories.\n"
        "2. For each piece of information you identify, explain why it is or isn’t considered private.\n"
        "3. Based on your analysis, conclude whether the message contains private information. Respond with 'Yes' if private data is present, "
        "or 'No' if it is not.\n\n"
        f"Message: '{message}'\n"
        "Let's think step by step."
    )
    response = llama_model.invoke([HumanMessage(content=private_data_prompt)]).content
    response = response.strip().lower()
    print(response)
    return "yes" in response


# check message clarity 
def check_message_clarity(message: str, llm: Callable) -> bool:
    """Check if the message is clear using the LLM."""
    clarity_prompt = (
        f"Analyze the following message to determine if it is clear and provides enough information to respond accurately. "
        f"Follow these steps to reach your decision:\n"
        f"1. Identify the main intent or purpose of the message. State what the user is asking or communicating.\n"
        f"2. Check if all necessary information is present to provide a complete response. "
        f"If there is missing or ambiguous information, note what is unclear.\n"
        f"3. Conclude with 'Yes' if the message is clear and sufficient for you to respond appropriately, or 'No' if it is ambiguous and requires clarification.\n"
        f"4. If you conclude with 'No', provide a specific clarifying question to obtain the missing information.\n\n"
        f"Message: '{message}'\n"
        f"Let's think step by step."
    )
    # Get the LLM's response
    response = llm.invoke([HumanMessage(content=clarity_prompt)]).content.strip().lower()
    # Check the LLM's response for 'Yes' or 'No'
    print( f"LLM response: {response}")
    if "need more information" in response or "clarify" in response or "unclear" in response:
        return False
    return True


# %%
from typing import Callable

# def check_message_clarity(message: str, llm: Callable) -> bool:
#     """Check if the message is clear using the LLM."""
#     clarity_prompt = (
#         f"The following message may contain an incomplete or unclear request. "
#         f"Your task is to understand the intent of the message as much as possible. "
#         f"If the message is clear, provide an appropriate response. "
#         f"However, if the message is ambiguous or you need more information to respond accurately, "
#         f"ask specific, clarifying questions to gather the necessary details. "
#         f"Ensure your questions are direct and focused on obtaining the missing information needed to proceed effectively."
#         f"\n\nMessage: '{message}'"
#     )
    
#     # Get the LLM's response
#     response = llama_model.invoke([HumanMessage(content=clarity_prompt)]).content
#     # Check the LLM's response for 'Yes' or 'No'
#     print( f"LLM response: {response}")
#     response_cleaned = response.strip().lower()
#     if "need more information" in response or "clarify" in response or "unclear" in response:
#         return False
#     return True

def check_private_data_with_llm(message: str, llama_model: Callable) -> bool:
    """Check if the message contains private data using the local LLM."""
    # Create a prompt for the LLM
    prompt = (
            f"Analyze the following message to determine if it contains private or sensitive information, such as personal identifiers "
            f"(e.g., social security numbers, passport numbers), contact information (e.g., phone numbers, email addresses), "
            f"financial details (e.g., credit card numbers, bank account information), or personal health information.\n\n"
            f"Message: '{message}'\n\n"
            f"Please respond with 'True' if the message contains private or sensitive information, or 'False' if it does not."
        )
    
    # Get the LLM's response
    response = llama_model.invoke([HumanMessage(content=prompt)]).content
    # Check the LLM's response for 'Yes' or 'No'
    print( f"LLM response: {response}")
    response_cleaned = response.strip().lower()
    if "true" in response_cleaned or "yes" in response_cleaned :
        return True
    elif "no" in response_cleaned:
        return False
    else :
        # If the response is unclear, assume it may contain private data
        return True
    
def process_message(message: str, local_llm: Callable) -> str:
    """Process the message based on whether it contains private data or needs clarification."""
    if check_private_data_with_llm(message, local_llm):
        print("Private data detected. Using local agent.")
        return "Local"
    else:
        print("Message is clear and does not contain private data. Using public agent.")
        return "Public"


# %%
def process_messages_from_file(filename, llama_model, llama_agent, claude_agent):
    try:
        with open(filename, 'r') as file:
            messages = file.readlines()
            for user_message in messages:
                user_message = user_message.strip()
                if not user_message:
                    continue
                
                response = process_message(user_message, llama_model)
                if response == "Local":
                    config = {"configurable": {"thread_id": "xyz123"}}
                    for chunk in llama_agent.stream(
                        {"messages": [HumanMessage(content=user_message)]}, config
                    ):
                        print(chunk)
                        print("----")
                else:
                    config = {"configurable": {"thread_id": "xyz456"}}
                    for chunk in claude_agent.stream(
                        {"messages": [HumanMessage(content=user_message)]}, config
                    ):
                        print(chunk)
                        print("----")

    except FileNotFoundError:
        print("The file was not found. Please check the filename and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")

# %%
# config = {"configurable": {"thread_id": "xyz456"}}
# temp = "what are my events for tommorrow"
# def parse_chunk(chunk):
#     if 'agent' in chunk:
#         agent_messages = chunk['agent']['messages']
#         for message in agent_messages:
#             print("Agent Message:")
#             print(message.content)

#     if 'tools' in chunk:
#         tool_messages = chunk['tools']['messages']
#         for message in tool_messages:
#             print("Tool Message:")
#             print(f"Tool: {message.name}")
#             print(f"Content: {message.content}")
#             print("----")

# # Use this function in your loop
# # response = claude_agent.invoke(
# #     {"messages": [HumanMessage(content="whats the weather in austin?")]}, config
# # )

# for chunk in claude_agent.stream(ls

#     {"messages": [HumanMessage(content="hi im bob!")]}, config
# ):
#     print(chunk)
#     print("----")

# %%

while True:
    user_message = input("Enter your message (type 'quit' to exit): ")
    if user_message.lower() == 'quit':
        break
    elif user_message.lower() == 'load':
        filename = "messages.txt"
        process_messages_from_file(filename, llama_model, llama_agent, claude_agent)
    else :
    
        response = process_message(user_message, llama_model)
        if response == "Local":
            config = { "configurable": {"thread_id": "xyz123"} }
            for chunk in llama_agent.stream(
                {"messages": [HumanMessage(content=user_message)]}, config
            ):
                print(chunk)
                print("----")
        else:
            config = {"configurable": {"thread_id": "xyz456"}}
            for chunk in claude_agent.stream(
                {"messages": [HumanMessage(content=user_message)]}, config
            ):
                print(chunk)
                print("----")


