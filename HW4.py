
import os
from dotenv import load_dotenv
import os

import sys
import io

# Step 1: Redirect stdout and stderr
original_stdout = sys.stdout  # Save a reference to the original standard output
original_stderr = sys.stderr  # Save a reference to the original standard error
sys.stdout = io.StringIO()    # Redirect standard output to a StringIO object
sys.stderr = io.StringIO()  

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

import subprocess
import time
import ollama

def stop_ollama_processes():
    try:
        # Find all running Ollama processes
        result = subprocess.run(["pgrep", "-f", "ollama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            pids = result.stdout.decode().splitlines()
            for pid in pids:
                # Terminate each process
                subprocess.run(["kill", "-15", pid])
            print("All Ollama processes have been stopped.")
            # Wait a moment to ensure processes have time to terminate
            time.sleep(2)
        else:
            print("No running Ollama processes found.")
    except Exception as e:
        print(f"Error stopping Ollama processes: {e}")

def start_ollama_server():
    try:
        # Start the Ollama server
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Ollama server started successfully.")
        return process
    except Exception as e:
        print(f"Error starting Ollama server: {e}")
        return None

def is_ollama_running():
    try:
        # Try to list models, if successful, Ollama is running
        ollama.list()
        return True
    except Exception:
        return False

# Start the Ollama server
ollama_process = start_ollama_server()

# Wait for the server to start
while not is_ollama_running():
    time.sleep(1)
    print("Waiting for Ollama server to start...")

print("Ollama server is running.")



 
# Set up Local LLM
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

llama_model = ChatOllama(model="llama3.2")

 
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
    Action.GOOGLECALENDAR_UPDATE_EVENT,
    Action.GOOGLECALENDAR_FIND_EVENT,
    Action.GOOGLECALENDAR_QUICK_ADD,
    Action.GOOGLECALENDAR_GET_CURRENT_DATE_TIME
])

 
# Setup Internet search tool
from langchain_community.tools import DuckDuckGoSearchResults

internet_search_tool = DuckDuckGoSearchResults()

 
import os
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from typing import Dict, Any

class PDFQASystem:
    def __init__(self):
        self.pdf_dir = "pdf_documents"
        os.makedirs(self.pdf_dir, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.vectorstore = Chroma(embedding_function=self.embeddings, persist_directory="./chroma_db")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=claude_model,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        # Add a set to keep track of added PDFs
        self.added_pdfs = set()

    def check_if_pdf_valid(self, file_path: str) -> bool:
        return os.path.exists(file_path) and file_path.lower().endswith('.pdf')

    def add_pdf(self, file_path):
        # Check if the PDF has already been added
        if file_path in self.added_pdfs:
            print(f"PDF '{file_path}' has already been added to the system.")
            return True

        if not self.check_if_pdf_valid(file_path):
            print(f"Error: '{file_path}' is not a valid PDF file or does not exist.")
            return False

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            splits = self.text_splitter.split_documents(documents)
            self.vectorstore.add_documents(splits)
            
            # Add the file path to the set of added PDFs
            self.added_pdfs.add(file_path)
            
            print(f"Added {file_path} to the system.")
            return True
        except Exception as e:
            print(f"Error adding PDF '{file_path}': {str(e)}")
            return False
        
    def format_output(self, result: Dict[str, Any]) -> str:
        formatted = "=" * 50 + "\n"
        formatted += f"Question: {result['question']}\n"
        formatted += "=" * 50 + "\n\n"
        
        formatted += "Answer:\n"
        formatted += "-" * 50 + "\n"
        formatted += f"{result['answer']}\n\n"
        formatted += "-" * 50 + "\n"
        
        formatted += "Sources:\n"
        formatted += "-" * 50 + "\n"
        for source in result['sources'].split(", "):
            formatted += f"- {source}\n"
        formatted += "-" * 50 + "\n"

        return formatted
    def query(self, question):
        result = self.qa_chain({"question": question})
        print(self.format_output(result))
        return None

# Usage
qa_system = PDFQASystem()


 
# Combine all tools
tools = toolkit.get_tools()
tools.extend(google_calendar_tools)
llama_tools = tools.copy()
tools.append(internet_search_tool)
claude_tools = tools.copy()
# tools.append(pdf_qa_tool)

 
# Set up agent with claude model

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

claude_memory = MemorySaver() # Initialize memory saver

claude_agent = create_react_agent(claude_model, claude_tools, checkpointer=claude_memory)

 
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print("Tool Called")
        
        elif isinstance(message, dict) and 'text' in message:
            # Handle AI text response
            print("AI Response:")
            print(message['text'])
        
        elif isinstance(message, dict) and message.get("type") == "tool_use":
            # Handle tool use
            tool_name = message.get("name", "Unknown Tool")
            query = message.get("input", {}).get("query", "No query provided")
            print(f"Tool Called: {tool_name}")
            print(f"Query: {query}")
        
        else:
            message.pretty_print()


 
# Set up agent with llama model
from langchain_core.messages import SystemMessage

llama_memory = MemorySaver() # Initialize memory saver

llama_agent = create_react_agent(llama_model, llama_tools, checkpointer=llama_memory)#, state_modifier=SystemMessage)

 
llama_agent.verbose = True


 
from typing import Callable
from langchain.schema import HumanMessage
from typing import Callable
from langchain_core.messages import HumanMessage
import re

def process_message(message: str, local_llm: Callable) -> str:
    """Process the message to determine which agent to use based on private data, vague queries, or PDF-related questions."""
    if "<private>" in message.lower() or "<sensitive>" in message.lower() or "<confidential>" in message.lower():
        print("Private data detected. Using local agent.")
        return "Local"
    if "<pdf>" in message.lower() or "<document>" in message.lower(): 
        print("PDF-related query detected. Using PDF agent.")
        return "PDF"
    if "<public>" in message.lower() or "<email>" in message.lower() or "<contact>" in message.lower():
        print("Public query detected. Using public agent.")
        return "Public"
    
    prompt = (
        "Please analyze the following message step by step to determine its classification based on specific criteria. "
        "Go through each step and follow the instructions carefully. Your response should end with exactly one tag: `<PRIVATE>`, `<VAGUE>`, `<PDF>`, or `<PUBLIC>`.\n\n"
        
        "Message Analysis:\n"
        f"Message: '{message}'\n\n"

        "Step-by-Step Evaluation:\n\n"
        
        "Step 1: Privacy Check\n"
        "- Question: Does the message contain any private or sensitive data, such as personal identifiers, financial information, or health details?\n"
        "- Answer: If yes, the message should be classified as `<PRIVATE>`. If not, move to the next step.\n\n"
        
        "Step 2: Clarity Assessment\n"
        "- Question: Is the message vague or unclear, lacking sufficient detail or context to be fully understood?\n"
        "- Answer: If yes, the message should be classified as `<VAGUE>`. If not, continue to the next step.\n\n"
        
        "Step 3: PDF Relevance\n"
        "- Question: Does the message specifically pertain to PDFs, such as a request to retrieve, search, or answer questions based on PDF content?\n"
        "- Answer: If yes, classify the message as `<PDF>`. If not, proceed to the next step.\n\n"
        
        "Step 4: Final Classification\n"
        "- Question: If none of the previous tags apply, does the message lack private data, is it clear, and is not PDF-related?\n"
        "- Answer: If yes, classify the message as `<PUBLIC>`.\n\n"
        
        "Final Instruction:\n"
        "After considering each step carefully, respond with only the classification tag: `<PRIVATE>`, `<VAGUE>`, `<PDF>`, or `<PUBLIC>`.\n"
    )
    print("Classifying message...")
    # Get the LLM's response
    response = local_llm.invoke([HumanMessage(content=prompt)])
    # .content
    response.pretty_print()
    response = response.content.strip()
    
    # Extract only the final standalone tag
    matches = re.findall(r"<(PRIVATE|VAGUE|PDF|PUBLIC)>", response, re.IGNORECASE)
    if matches:
        classification = f"<{matches[-1].upper()}>"
        print(f"Detected classification: {classification}")
        
        if classification == "<PRIVATE>":
            return "Local"
        elif classification == "<VAGUE>":
            return "Vague"
        elif classification == "<PDF>":
            return "PDF"
        else:
            return "Public"
    else:
        print("No valid tag detected. Defaulting to Public.")
        return "Public"



 

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

# Agent Configuration
def get_agent_config(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}

def extract_pdf_paths(user_message):
    # Regular expression to identify PDF paths in the user's message
    pdf_paths = re.findall(r'\b(\S+\.pdf)\b', user_message)
    return pdf_paths

def get_user_input():
    placeholder_text = HTML('<style fg="gray">Send a message (or /bye for exit)</style>')
    # Display the placeholder, which disappears on typing
    user_input = prompt('\n>> ', placeholder=placeholder_text)
    return user_input

# Restore stdout and stderr
sys.stdout = original_stdout
sys.stderr = original_stderr

# Main interaction loop
while True:
    user_message = get_user_input()
    if user_message.lower() == 'quit' or user_message.lower() == '/bye' or user_message.lower() == 'exit':
        break

    response_type = process_message(user_message, llama_model)
    message = {"messages": [('user', user_message)]}
    
    if response_type == "Local":
        config = get_agent_config("xyz123")
        try:
            print_stream(llama_agent.stream(message, stream_mode="values", config=config))
        except Exception as e:
            print(f"Error with Local agent: {e}")
    
    elif response_type == "Vague":
        print("Your query was unclear. Please provide more detail.")
    
    elif response_type == "PDF":
        pdf_files = extract_pdf_paths(user_message)
        try:
            # Use the PDF QA system
            status = True
            for file in pdf_files:
                status = qa_system.add_pdf(file)
                if status == False:
                    break
            if status == False:
                continue
            result = qa_system.query(user_message)
        except Exception as e:
            print(f"Error with PDF QA system: {e}")
    
    else:
        config = get_agent_config("xyz456")
        try:
            print_stream(claude_agent.stream(message, stream_mode="values", config=config))
        except Exception as e:
            print(f"Error with Public agent: {e}")


