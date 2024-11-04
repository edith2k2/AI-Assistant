# %%
# Standard library imports
import logging
import sys
import os
import pickle
import base64
import re
import time
import random
import builtins
import operator
from datetime import datetime, timedelta
from email.message import EmailMessage
import json

# Third-party imports
from typing import List, Optional, Dict, Union, Type, Any, Literal, TypedDict
from typing_extensions import Annotated
from pydantic import BaseModel, Field

# Date/Time related
from dateutil import parser, tz
import pytz
import dateparser

# Google API related
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from google.auth.exceptions import RefreshError

# LangChain related
from langchain_core.tools import BaseTool, tool
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.output_parsers import PydanticOutputParser
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

# LangGraph related
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Prompt toolkit related
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.output import Output
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit import print_formatted_text as print_formatted
from prompt_toolkit.shortcuts import clear
# Environment variables
from dotenv import load_dotenv

# rich library imports
from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.console import Group
from rich._log_render import LogRender
from rich.padding import Padding
import shutil


class Colors:
    """ANSI color codes"""
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"

load_dotenv(override=True)  # This loads the .env file in the current directory

# Verify that the environment variable is loaded
api_key = os.getenv("ANTHROPIC_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if api_key is None:
    raise ValueError(f"{Colors.RED}API key is not set. Please set the 'ANTHROPIC_API_KEY' environment variable in .env and unset any other previous environment key.{Colors.END}")
print(f"{Colors.GREEN}API key loaded successfully.{Colors.END}")

# Get terminal size
terminal_width, terminal_height = shutil.get_terminal_size()

# %%
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly"
]

    
# Create a custom theme for consistent styling
custom_theme = Theme({
    'info': 'cyan',
    'warning': 'yellow',
    'error': 'red',
    'success': 'green',
    'stream': 'bright_blue'
})

# Initialize Rich console
console = Console(
    theme=custom_theme,
    soft_wrap=True,
    markup=True
)

def stream_print(text: str, delay: float = 0.02, end: Optional[str] = None, color: str = Colors.LIGHT_WHITE) -> None:
    """Print text with streaming effect and color.
    
    Args:
        text (str): The text to print
        delay (float): Delay between each character in seconds
        end (str, optional): String to print at the end
        color (str): ANSI color code to use
    """
    if not text:
        return

    sys.stdout.write(color)  # Set color
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(Colors.END)  # Reset color

    if end is not None:
        sys.stdout.write(end)
        sys.stdout.flush()

def smart_output(text: str, color: str = Colors.LIGHT_WHITE, stream_threshold: int = 100) -> None:
    """Smart output that decides between streaming and batch printing.
    
    Args:
        text (str): Text to output
        color (str): ANSI color code
        stream_threshold (int): Character length threshold for streaming
    """
    if not text:
        return

    # For short responses or user interaction, use streaming
    if len(text) < stream_threshold or any(marker in text.lower() for marker in 
        ['processing', 'thinking', 'please wait', 'loading', '...']):
        stream_print(text, color=color)
        return
    print("smart output")

    # For data-heavy content, use batch printing with progress indicator
    segments = text.split('\n')
    total_segments = len(segments)
    
    # Show processing indicator
    stream_print("Processing output", color=Colors.LIGHT_PURPLE)
    for _ in range(3):
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.2)
    print("\n")
    
    # Print content in visually distinct chunks
    for i, segment in enumerate(segments, 1):
        if segment.strip():  # Only process non-empty lines
            if any(header in segment.lower() for header in ['subject:', 'body:', 'results:', 'summary:']):
                # Headers get special formatting
                print(f"{Colors.YELLOW}{segment.strip()}{Colors.END}")
            else:
                # Regular content
                print(f"{color}{segment.strip()}{Colors.END}")
            
            # Add slight delay between major sections for readability
            if i < total_segments and i % 5 == 0:
                time.sleep(0.1)

# Configure logging
def setup_logger():
    """Configure logging with file handler only to avoid double printing"""
    logger = logging.getLogger('workflow_logger')
    logger.setLevel(logging.DEBUG)
    
    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a file handler
    file_handler = logging.FileHandler('workflow.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add only file handler to avoid console duplication
    logger.addHandler(file_handler)
    
    # Prevent propagation to the root logger
    logger.propagate = False
    
    # Suppress other loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logger


# Use the logger
logger = setup_logger()




def create_prompt_session():
    """
    Create a prompt session with a shadow placeholder that maintains scrollability.
    """
    # Define custom style with more contrasting colors
    style = Style.from_dict({
        'prompt': 'bold cyan',                    # The "You: " part
        'placeholder': 'italic #666666',          # Placeholder text style
        'continuation': 'gray',                   # Style for long input continuation
        'bottom-toolbar': 'bg:#222222 #ffffff',   # Bottom toolbar style
    })

    # Create the session with proper configuration
    session = PromptSession(
        message=[('class:prompt', 'You: ')],
        style=style,
        placeholder='type your message or "help" for usage guide...',
        enable_history_search=True,
        mouse_support=True,
        complete_while_typing=True,
        wrap_lines=True,
        # This ensures proper terminal handling
        input_processors=None,
        # Maintain scrollable history
        multiline=False,
        # Keep prompt at bottom
        bottom_toolbar=HTML('<b>Ctrl+C</b> to cancel  <b>↑↓</b> for history  <b>Tab</b> to complete')
    )
    return session
# %%


# %%


class PrivacyCheck(BaseModel):
    """Result of privacy check on user message."""
    contains_private_info: bool = Field(description="Whether the message contains private information")
    explanation: str = Field(description="Brief explanation of the decision")

class TaskType(BaseModel):
    """Determined task type from user message."""
    task: Literal["email", "calendar", "pdf", "search"] = Field(description="The determined task type")
    explanation: str = Field(description="Brief explanation of why this task type was chosen")

class EmailFields(TypedDict):
    to: str
    subject: str
    body: str
    cc: Optional[List[str]]
    bcc: Optional[List[str]]


class CalendarFields(BaseModel):
    summary: Optional[str] = Field(
        default=None,
        description="The title or brief description of the calendar event"
    )
    
    start_datetime: Optional[str] = Field(
        default=None,
        description="The start date and time of the event in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    
    end_datetime: Optional[str] = Field(
        default=None,
        description="The end date and time of the event in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    
    location: Optional[str] = Field(
        default=None,
        description="The location where the event will take place"
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Additional details or notes about the event"
    )


class PDFQASystem:
    def __init__(self, is_private: bool = False):
        """Initialize the PDF QA system"""
        self.pdf_dir = "pdf_documents"
        os.makedirs(self.pdf_dir, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        self.is_private = is_private

        self.llm = ChatOllama(model="llama3.2") if self.is_private else ChatAnthropic(model_name="claude-3-sonnet-20240229")
        
        # Initialize empty FAISS index
        self.vectorstore = FAISS.from_texts(
            ["Initial document"], 
            self.embeddings
        )
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Track added PDFs and their content
        self.added_pdfs = {}

    def check_if_pdf_valid(self, file_path: str) -> bool:
        """Check if the file exists and is a PDF"""
        return os.path.exists(file_path) and file_path.lower().endswith('.pdf')

    def add_pdf(self, file_path: str) -> bool:
        """Add a PDF to the system"""
        # Check for duplicate
        if file_path in self.added_pdfs:
            print(f"{Colors.YELLOW}PDF '{file_path}' has already been added to the system.{Colors.END}")
            return True

        # Validate PDF
        if not self.check_if_pdf_valid(file_path):
            print(f"{Colors.RED}Error: '{file_path}' is not a valid PDF file or does not exist.{Colors.END}")
            return False

        try:
            # Load and process the PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add source and page metadata
            for doc in documents:
                doc.metadata["source"] = file_path
            
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            
            # Store the splits for this PDF
            self.added_pdfs[file_path] = splits
            
            # Recreate the vector store with all documents
            all_splits = []
            for pdf_splits in self.added_pdfs.values():
                all_splits.extend(pdf_splits)
            
            self.vectorstore = FAISS.from_documents(all_splits, self.embeddings)
            
            # Reinitialize QA chain with updated vector store
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            print(f"{Colors.LIGHT_GREEN}Added {file_path} to the system.{Colors.END}")
            return True
            
        except Exception as e:
            print(f"{Colors.LIGHT_RED}Error adding PDF '{file_path}': {str(e)} {Colors.END}")
            return False

    def extract_pdf_paths(self, user_message: str) -> List[str]:
        """Extract PDF paths from angle brackets in the message"""
        pattern = r'<([^>]+\.pdf)>'
        return re.findall(pattern, user_message, re.IGNORECASE)

    def format_output(self, result: Dict[str, Any]) -> str:
        """Format the output similarly to the original"""
        formatted = "=" * 50 + "\n"
        formatted += f"Question: {result['question']}\n"
        formatted += "=" * 50 + "\n\n"
        
        formatted += "Answer:\n"
        formatted += "-" * 50 + "\n"
        formatted += f"{result['answer']}\n\n"
        formatted += "-" * 50 + "\n"
        
        formatted += "Sources:\n"
        formatted += "-" * 50 + "\n"
        
        # Extract sources from source_documents
        if 'source_documents' in result:
            for doc in result['source_documents']:
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'Unknown page')
                formatted += f"- {source} (Page {page + 1})\n"
        
        formatted += "-" * 50 + "\n"
        return formatted

    def query(self, question: str) -> Dict[str, Any]:
        """Query the system and return results"""
        if not self.added_pdfs:
            return {
                "question": question,
                "answer": f"{Colors.YELLOW}No documents have been added to the system yet.{Colors.END}",
                "source_documents": []
            }
            
        try:
            # result = self.qa_chain({"query": question})
            
            # # Format the result for consistency
            # formatted_result = {
            #     "question": question,
            #     "answer": result.get("result", "No answer found."),
            #     "source_documents": result.get("source_documents", [])
            # }
            result = self.qa_chain.invoke({
                "query": question,
                "chat_history": []  # Add if you're tracking chat history
            })
            
            # Format the result for consistency
            formatted_result = {
                "question": question,
                "answer": result.get("result", "No answer found."),
                "source_documents": result.get("source_documents", [])
            }
            
            # print(self.format_output(formatted_result))
            stream_print(self.format_output(formatted_result), color=Colors.CYAN)
            return formatted_result
            
        except Exception as e:
            error_result = {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "source_documents": []
            }
            stream_print(self.format_output(error_result), color=Colors.LIGHT_RED)
            return error_result


# %%
class PDFState(TypedDict):
    file_path: List[str]
    question: str
    is_valid: bool
    is_processed: bool
    answer: Optional[str]
    sources: Optional[str]
    error: Optional[str]
    qa_system: Optional[PDFQASystem]  

# %%
# Define the agent state and other states

# Define our states


class EmailState(TypedDict):
    to: str
    cc: str
    bcc: str
    subject: str
    body: str
    additional_instructions: str
    missing_fields: List[str]

class CalendarState(BaseModel):
    summary: str = Field(
        default="",
        description="The title or brief description of the calendar event"
    )
    
    start_datetime: str = Field(
        default="",
        description="The start date and time of the event in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    
    end_datetime: str = Field(
        default="",
        description="The end date and time of the event in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    
    location: Optional[str] = Field(
        default="",
        description="The location where the event will take place (optional)"
    )
    
    description: Optional[str] = Field(
        default="",
        description="Additional details or notes about the event (optional)"
    )
    
    timezone: str = Field(
        default="America/Chicago",
        description="The timezone in which the event times are specified"
    )
    
    missing_fields: List[str] = Field(
        default_factory=list,
        description="A list of field names that are required but currently missing or empty"
    )

    changes_requested: List[str] = Field(
        default_factory=list,
        description="A list of changes requested by the user"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if any error occurred during event creation"
    )

    event_result: Optional[str] = Field(
        default=None,
        description="Result message after successful event creation"
    )



class SearchState(TypedDict):
    query: str
    search_results: Optional[str]
    final_response: Optional[str]
    error: Optional[str]

class LLMState(BaseModel):
    private_llm: Optional[ChatOllama] = None
    public_llm: Optional[ChatAnthropic] = None
    
    def get_llm(self, is_private: bool = False):
        """Get appropriate LLM instance based on privacy setting"""
        if is_private:
            if self.private_llm is None:
                self.private_llm = ChatOllama(model="llama3.2")
            return self.private_llm
        else:
            if self.public_llm is None:
                self.public_llm = ChatAnthropic(model_name="claude-3-sonnet-20240229")
            return self.public_llm

class AgentState(TypedDict):
    task_type: TaskType
    contains_private_info: bool
    user_message: str 
    email: EmailState | None
    calendar: CalendarState | None
    pdf: PDFState | None
    search: SearchState | None
    approved: bool
    changes_requested: Annotated[List[str], operator.add]
    llm_state: LLMState 





# Create the workflow



# %%
def parse_email_response(response: str) -> dict:
    """
    Parse the response from email_chain.run() to extract subject and body.
    
    Args:
    response (str): The raw response string from email_chain.run()
    
    Returns:
    dict: A dictionary containing 'subject' and 'body' keys with their respective values
    """
    # Initialize variables
    subject = ""
    body = ""
    current_section = None

    # Split the response into lines
    logger.debug(f"Parsing email response: {response}")
    text = response.content
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        
        # Check for section headers
        if line.startswith("SUBJECT:"):
            current_section = "subject"
            subject = line[8:].strip()  # Remove "SUBJECT:" and any leading/trailing whitespace
        elif line.startswith("BODY:"):
            current_section = "body"
            continue  # Skip the "BODY:" line itself
        
        # Add content to the appropriate section
        elif current_section == "subject":
            subject += " " + line
        elif current_section == "body":
            body += line + "\n"

    # Remove any trailing newline from the body
    body = body.rstrip()

    return {
        "subject": subject,
        "body": body
    }


# %%
# email functions 

def build_gmail_service():
    logger.debug("Getting Gmail service.")
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('email_token.json'):
        with open('email_token.json', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                # If refresh fails, we'll fall through to the else block and get new credentials
                creds = None
        
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', ["https://mail.google.com/"])
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('email_token.json', 'wb') as token:
            pickle.dump(creds, token)
    logger.debug("Gmail service obtained successfully.")

    return build('gmail', 'v1', credentials=creds)

# Email handling functions
def check_email_missing_fields(state: AgentState) -> AgentState:
    # Initialize email state if it doesn't exist
    logger.debug("Checking for missing email fields.")
    if state['email'] is None:
        state['email'] = {}
    
    missing = []
    
    # Check for missing fields
    if not state['email'].get('to'):
        missing.append('to')
    if not state['email'].get('subject'):
        missing.append('subject')
    if not state['email'].get('body'):
        missing.append('body')
    
    # Optionally check for CC and BCC
    if 'cc' in state['email'] and not state['email']['cc']:
        missing.append('cc')
    if 'bcc' in state['email'] and not state['email']['bcc']:
        missing.append('bcc')
    
    logger.debug(f"Missing fields: {missing}")
    # Update the state with the list of missing fields
    if missing:
        state['email']['missing_fields'] = missing
    logger.debug(f"Updated state with missing fields: {state['email']}")
    return state


def ask_email_missing_fields(state: AgentState) -> AgentState:
    logger.debug("Asking for missing email fields.")
    if state['email'] is None:
        raise ValueError("Email state is not initialized")
    missing_fields = []
    try :
        missing_fields = state['email']['missing_fields']
    except KeyError as e:
        raise ValueError(f"Missing fields not found in state: {str(e)}")

    if 'to' in missing_fields:
        sys.stdout.write(f"\n{Colors.YELLOW}Please provide the email address of the recipient:{Colors.END} ")
        sys.stdout.flush()
        recipient = input().strip()
        state['email']['to'] = recipient

    if 'subject' in missing_fields:
        sys.stdout.write(f"\n{Colors.YELLOW}Please provide a subject for the email:{Colors.END} ")
        sys.stdout.flush()
        subject = input().strip()
        state['email']['subject'] = subject

    if 'body' in missing_fields:
        sys.stdout.write(f"\n{Colors.YELLOW}Please provide the content for the email body:{Colors.END} ")
        sys.stdout.flush()
        body = input().strip()
        state['email']['body'] = body

    if 'cc' in missing_fields:
        cc = builtins.input("Please provide CC recipients (comma-separated email addresses, or press Enter if none): ")
        state['email']['cc'] = [email.strip() for email in cc.split(',')] if cc else []

    if 'bcc' in missing_fields:
        bcc = builtins.input("Please provide BCC recipients (comma-separated email addresses, or press Enter if none): ")
        state['email']['bcc'] = [email.strip() for email in bcc.split(',')] if bcc else []

    # Remove the 'missing_fields' key from the email state
    state['email'].pop('missing_fields', None)
    logger.debug(f"Updated email state after asking for missing fields: {state['email']}")

    return state


def draft_email(state: AgentState) -> AgentState:
    logger.debug("Drafting email.")
    if state['email'] is None:
        raise ValueError("Email state is not initialized")
    
    llm = state['llm_state'].get_llm(is_private=state['contains_private_info'])
    # Initialize ChatOllama
    # llm = ChatOllama(model="llama3.2")

    # Prepare the prompt for ChatOllama
    email_prompt = PromptTemplate(
        input_variables=["recipient", "subject", "cc", "bcc", "body", "additional_instructions"],
        template="""
You are writing a professional email. Use the following information:
To: {recipient}
Subject: {subject}
CC: {cc}
BCC: {bcc}
Key points: {body}
Instructions:
1. Start with "SUBJECT:" followed by an expanded, professional subject line.
2. On a new line, start with "BODY:" followed by the full email body.
3. Expand on the key points to create a complete, professional email.
4. Do not include any other text or explanations.
{additional_instructions}
Remember, only include the SUBJECT and BODY, nothing else.
""")
    
    prompt_input = {
        "recipient": state['email'].get('recipient', 'N/A'),
        "subject": state['email'].get('subject', 'N/A'),
        "cc": ', '.join(state['email'].get('cc', [])) or 'N/A',
        "bcc": ', '.join(state['email'].get('bcc', [])) or 'N/A',
        "body": state['email'].get('body', ''),
        "additional_instructions": state['email'].get('additional_instructions', '')
    }

    # Generate the email draft using LLM
    # email_chain = LLMChain(llm=llm, prompt=email_prompt)
    email_chain = email_prompt | llm
    response = email_chain.invoke(prompt_input)

    # # Generate the email draft using ChatOllama
    # email_chain = LLMChain(llm=llm, prompt=email_prompt)

    logger.debug("Email draft generated successfully.")
    # Parse the response
    parsed_email = parse_email_response(response)

    # Update the state with the parsed email
    state['email']['subject'] = parsed_email['subject']
    state['email']['body'] = parsed_email['body']
    logger.debug(f"Updated email state after drafting: {state['email']}")
    
    return state


def send_email_node(state: AgentState) -> AgentState:
    logger.debug("Sending email.")
    service = build_gmail_service()
    message = EmailMessage()
    
    # Assuming the email content is stored in the state
    email_data = state['email']
    
    message.set_content(email_data['body'])
    message["To"] = email_data['to']
    message["From"] = "2k2vamshi@gmailc.om"  # Replace with your email
    message["Subject"] = email_data['subject']

    # Add CC if present
    if 'cc' in email_data and email_data['cc'] !="":
        message["Cc"] = ", ".join(email_data['cc'])
    
    # Add BCC if present
    if 'bcc' in email_data and email_data['bcc'] !="":
        message["Bcc"] = ", ".join(email_data['bcc'])
    
    # Encode the message
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    create_message = {"raw": encoded_message}
    
    # Send the message
    send_message = service.users().messages().send(userId="me", body=create_message).execute()
    
    # You might want to update the state to indicate the email was sent
    state['email']['sent'] = True
    
    return state


def user_approval_and_changes(state: AgentState) -> AgentState:
    # print("Draft Email:")
    # print(f"Subject: {state['email'].get('subject', 'N/A')}")
    # print(f"Body: {state['email'].get('body', 'N/A')}")
    
    sys.stdout.write(f"\n{Colors.YELLOW}Do you approve this email? (yes/no):{Colors.END} ")
    sys.stdout.flush()
    approval = input().lower().strip()
    state['approved'] = approval == 'yes'
    
    if not state['approved']:
        sys.stdout.write(f"\n{Colors.YELLOW}Please specify the changes needed or provide additional instructions:{Colors.END} ")
        sys.stdout.flush()
        changes = input().strip()
        if 'additional_instructions' not in state['email']:
            state['email']['additional_instructions'] = ''
        state['email']['additional_instructions'] += f"\n{changes}" if state['email']['additional_instructions'] else changes
    
    return state

# %%
# search functions


search_tool = DuckDuckGoSearchRun()

def internet_search(state: AgentState) -> AgentState:
    """
    Perform internet search using DuckDuckGo
    """
    if state['search'] is None:
        # Initialize search state if not present
        state['search'] = SearchState(
            query=state['user_message'],
            search_results=None,
            final_response=None,
            error=None
        )
    
    try:
        # Perform the search
        results = search_tool.run(state['search']['query'])
        state['search']['search_results'] = results
        state['search']['error'] = None
    except Exception as e:
        state['search']['error'] = f"Search failed: {str(e)}"
        state['search']['search_results'] = None
        state['changes_requested'].append(f"Search error: {str(e)}")
    
    return state

class SearchResponse(BaseModel):
    answer: str = Field(description="The main answer synthesized from search results")
    confidence: float = Field(description="Confidence score between 0 and 1")
    sources: List[str] = Field(description="List of sources used in the answer")

def format_search_response(state: AgentState) -> AgentState:
    """
    Format the search results into a coherent response using LLM
    """
    if state['search'] is None:
        state['search'] = SearchState(
            query="",
            search_results=None,
            final_response="Error: No search state found",
            error="No search state initialized"
        )
        return state

    if state['search']['error']:
        state['search']['final_response'] = (
            f"I encountered an error while searching: {state['search']['error']}\n"
            "Would you like to try a different search query?"
        )
        return state
        
    if not state['search']['search_results']:
        state['search']['final_response'] = (
            "I couldn't find any results for your query. "
            "Would you like to try a different search term?"
        )
        return state

    # Select LLM based on privacy flag
    llm = state['llm_state'].get_llm(is_private=state['contains_private_info'])
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=SearchResponse)

    # Create prompt template with explicit format instructions
    prompt_template = """
    Based on the following search results, provide a response to the user's query.
    
    User Query: {query}
    
    Search Results:
    {search_results}
    
    Instructions:
    1. Directly address the user's query
    2. Synthesize information from the search results
    3. Structure the response clearly
    4. Include relevant details and context
    5. Note any limitations or uncertainties

    {format_instructions}
    
    Your response should be formatted exactly as specified above.
    """

    # Create prompt with parser instructions
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "search_results"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Create chain
    search_chain = prompt | llm | parser

    try:
        # Create a simpler prompt for the LLM to ensure consistent output
        prompt_template = """
        Based on the following search results, provide a comprehensive response.
        
        Query: {query}
        Search Results: {search_results}
        
        Provide your response in the following JSON format:
        {{
            "answer": "your detailed answer here",
            "confidence": 0.95,
            "sources": ["source1", "source2"]
        }}

        Requirements:
        1. The answer should be detailed and directly address the query
        2. Confidence should be between 0 and 1
        3. Include at least one source
        4. Format the response as valid JSON
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "search_results"]
        )

        # Call LLM with the prompt
        raw_response = llm.invoke(prompt.format(
            query=state['search']['query'],
            search_results=state['search']['search_results']
        ))

        # Try to extract JSON from the response
        content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
        
        # Try to find JSON in the content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            raise ValueError("No JSON found in response")
            
        json_str = json_match.group()
        processed_response = json.loads(json_str)

        # Format final response
        state['search']['final_response'] = f"""
Based on your query: "{state['search']['query']}"

{processed_response.get('answer', 'No answer found.')}

Confidence: {processed_response.get('confidence', 0.0):.2f}

Sources:
{', '.join(processed_response.get('sources', ['No sources available']))}

---
Note: This response was generated based on search results and may not be complete or fully accurate.
Would you like to know more about any specific aspect?
"""

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        # Fallback to raw results
        state['search']['final_response'] = (
            f"I found some results but couldn't process them properly.\n\n"
            f"Raw search results:\n{state['search']['search_results']}\n\n"
            "Would you like to try reformulating your query?"
        )
    except Exception as e:
        logger.error(f"Error processing search results: {str(e)}")
        state['search']['final_response'] = (
            f"While I found some results, I encountered an error processing them.\n"
            f"Raw search results:\n{state['search']['search_results']}\n\n"
            "Would you like to try reformulating your query?"
        )

    return state

def add_search_nodes(workflow: StateGraph) -> StateGraph:
    """
    Add search-related nodes to the main workflow
    """
    # Add nodes
    workflow.add_node("internet_search", internet_search)
    workflow.add_node("format_search_response", format_search_response)
    
    # Add edges
    workflow.add_edge("internet_search", "format_search_response")
    workflow.add_edge("format_search_response", END)
    
    return workflow

# %%
pdf_qa_system = None

def get_or_create_pdf_system(is_private: bool) -> PDFQASystem:
    """Get existing PDFQASystem or create new one if needed"""
    global pdf_qa_system
    
    if pdf_qa_system is None:
        pdf_qa_system = PDFQASystem(is_private=is_private)
    elif pdf_qa_system.is_private != is_private:
        # If privacy setting changed, create new system but preserve added PDFs
        old_pdfs = pdf_qa_system.added_pdfs
        pdf_qa_system = PDFQASystem(is_private=is_private)
        # Re-add existing PDFs to new system
        for pdf_path in old_pdfs:
            pdf_qa_system.add_pdf(pdf_path)
            
    return pdf_qa_system
def validate_pdf(state: AgentState) -> AgentState:
    """Validate PDF file and initialize PDF state"""
    if state['pdf'] is None:
        state['pdf'] = PDFState(
            file_path=[],
            question="",
            is_valid=False,
            is_processed=False,
            answer=None,
            sources=None,
            error=None,
            qa_system=get_or_create_pdf_system(state['contains_private_info'])  # Initialize QA system when PDF state is created
        )
    
    # Extract file path and question from user message
    added_pdfs = state['pdf']['qa_system'].added_pdfs
    # print("added_pdfs", added_pdfs)
    pdf_paths = state['pdf']['qa_system'].extract_pdf_paths(user_message=state['user_message'])

    if not pdf_paths and added_pdfs:
        state['pdf']['question'] = state['user_message']
        state['pdf']['is_valid'] = True
        state['pdf']['is_processed'] = True
        return state
    
    if not added_pdfs and not pdf_paths:
        state['pdf']['error'] = "No PDF file paths found in the message or database"
        print(f"{Colors.LIGHT_RED}No PDF file paths found in the message or database{Colors.END}")
        state['pdf']['question'] = state['user_message']
        state['pdf']['is_valid'] = False
        return state
    
    for pdf in pdf_paths:
        if not state['pdf']['qa_system'].check_if_pdf_valid(file_path=pdf):
            print(f"{Colors.LIGHT_RED}{pdf} is not a valid PDF file{Colors.END}")
            state['pdf']['error'] = "{pdf} is not a valid PDF file"
            state['pdf']['question'] = state['user_message']
            state['pdf']['is_valid'] = False
            return state
    state['pdf']['file_path'] = pdf_paths
    state['pdf']['question'] = state['user_message']
    state['pdf']['is_valid'] = True
    return state
    
def process_pdf(state: AgentState) -> AgentState:
    """Process PDF and add to the QA system"""
    if not state['pdf']['is_valid']:
        return state
    
    try:
        for pdf in state['pdf']['file_path']:
            success = state['pdf']['qa_system'].add_pdf(pdf)
            if not success:
                state['pdf']['error'] = "Error while processing PDF {pdf}"
                state['pdf']['question'] = state['user_message']
                state['pdf']['is_valid'] = False
                return state
        state['pdf']['is_processed'] = True
    except Exception as e:
        state['pdf']['error'] = f"Error processing PDF: {str(e)}"
        state['pdf']['is_processed'] = False
    
    return state

def answer_question(state: AgentState) -> AgentState:
    """Generate answer for the question"""
    if not state['pdf']['is_processed']:
        return state
    
    try:
        question = re.sub(r'<[^>]+\.pdf>', '', state['pdf']['question']).strip()
        result = state['pdf']['qa_system'].query(question)
        
        state['pdf']['answer'] = result['answer']
        state['pdf']['sources'] = result['sources']
        state['pdf']['error'] = None
    except Exception as e:
        state['pdf']['error'] = f"Error generating answer: {str(e)}"
        state['changes_requested'].append(state['pdf']['error'])
    
    return state

def format_pdf_response(state: AgentState) -> AgentState:
    """Format the final response"""
    if state['pdf']['error']:
        return state
    
    try:
        formatted_result = {
            'question': state['pdf']['question'],
            'answer': state['pdf']['answer'],
            'sources': state['pdf']['sources']
        }
        
        state['pdf']['answer'] = state['pdf']['qa_system'].format_output(formatted_result)
    except Exception as e:
        state['pdf']['error'] = f"Error formatting response: {str(e)}"
        state['changes_requested'].append(state['pdf']['error'])
    
    return state

def add_pdf_nodes(workflow: StateGraph) -> StateGraph:
    """Add PDF-related nodes to the main workflow"""
    
    # Add nodes
    workflow.add_node("validate_pdf", validate_pdf)
    workflow.add_node("process_pdf", process_pdf)
    workflow.add_node("answer_question", answer_question)
    workflow.add_node("format_pdf_response", format_pdf_response)
    
    # Add edges
    workflow.add_conditional_edges(
        "validate_pdf",
        lambda x: "process" if x['pdf']['is_valid'] else "end",
        {
            "process": "process_pdf",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "process_pdf",
        lambda x: "answer" if x['pdf']['is_processed'] else "end",
        {
            "answer": "answer_question",
            "end": END
        }
    )
    
    workflow.add_edge("answer_question", "format_pdf_response")
    workflow.add_edge("format_pdf_response", END)
    
    return workflow

# %%





class GoogleCalendarBaseTool(BaseTool):
    """Base class for Google Calendar tools."""
    
    api_resource: Resource = Field(default_factory=build_resource_service)
    
    @classmethod
    def from_api_resource(cls, api_resource: Resource) -> "GoogleCalendarBaseTool":
        """Create a tool from an api resource.

        Args:
            api_resource: The api resource to use.

        Returns:
            A tool.
        """
        logger.debug("Creating Google Calendar tool from API resource.")
        return cls(api_resource=api_resource)
    
# Create event tool
class CreateEventSchema(BaseModel):
    # https://developers.google.com/calendar/api/v3/reference/events/insert
    
    # note: modifed the tz desc in the parameters, use local time automatically
    start_datetime: str = Field(
        description=(
            " The start datetime for the event in the following format: "
            ' YYYY-MM-DDTHH:MM:SS, where "T" separates the date and time '
            " components, "
            ' For example: "2023-06-09T10:30:00" represents June 9th, '
            " 2023, at 10:30 AM"
            "Do not include timezone info as it will be automatically processed."
        )
    )
    end_datetime: str = Field(
        description=(
            " The end datetime for the event in the following format: "
            ' YYYY-MM-DDTHH:MM:SS, where "T" separates the date and time '
            " components, "
            ' For example: "2023-06-09T10:30:00" represents June 9th, '
            " 2023, at 10:30 AM"
            "Do not include timezone info as it will be automatically processed."
        )
    )
    summary: str = Field(
        description="The title of the event."
    )
    location: Optional[str] = Field(
        default="",
        description="The location of the event."
    )
    description: Optional[str] = Field(
        default="",
        description="The description of the event. Optional."
    )
    timezone: str = Field(
        default="America/Chicago",
        description="The timezone in TZ Database Name format, e.g. 'America/New_York'"
    )
    

class CreateGoogleCalendarEvent(GoogleCalendarBaseTool):
    name: str = "create_google_calendar_event"
    description: str = (
        " Use this tool to create a new calendar event in user's primary calendar."
        " The input must be the start and end datetime for the event, and"
        " the title of the event. You can also specify the location and description"
    )
    args_schema: Type[BaseModel] = CreateEventSchema
    
    def _run(
        self,
        start_datetime: str,
        end_datetime: str,
        summary: str,
        location: str = "",
        description: str = "",
        timezone: str = "America/Chicago",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
            
        # start = datetime.strptime(start_datetime, '%Y-%m-%dT%H:%M:%S')
        # start = start.replace(tzinfo=tz.gettz(timezone)).isoformat()
        # end = datetime.strptime(end_datetime, '%Y-%m-%dT%H:%M:%S')
        # end = end.replace(tzinfo=tz.gettz(timezone)).isoformat()
        start = parser.isoparse(start_datetime)
        end = parser.isoparse(end_datetime)

        # Convert to the specified timezone
        tz = pytz.timezone(timezone)
        start = start.astimezone(tz).isoformat()
        end = end.astimezone(tz).isoformat()
        
        calendar = 'primary' # default calendar
        body = {
            'summary': summary,
            'start': {
                'dateTime': start
            },
            'end': {
                'dateTime': end
            }
        }
        if location != "":
            body['location'] = location
        if description != "":
            body['description'] = description
        
        event = self.api_resource.events().insert(calendarId=calendar, body=body).execute()
        
        return "Event created: " + event.get('htmlLink', 'Failed to create event')
    
    async def _arun(
        self,
        start_datetime: str,
        end_datetime: str,
        summary: str,
        location: str = "",
        description: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        
        raise NotImplementedError("Async version of this tool is not implemented.")


def get_calendar_credentials(
    token_file: str,
    scopes: list,
    client_secrets_file: str
) -> Credentials:
    """
    Get or create Calendar credentials
    """
    creds = None
    
    # If token pickle exists, load it
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials available, create them
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes)
            creds = flow.run_local_server(port=0)
            
        # Save credentials for future use
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
            
    return creds

def build_calendar_service(credentials: Credentials):
    """
    Build and return the Calendar service
    """
    return build('calendar', 'v3', credentials=credentials)

# Usage:
SCOPES = ['https://www.googleapis.com/auth/calendar']

# Get credentials
credentials = get_calendar_credentials(
    token_file="./calendar_token.json",
    scopes=SCOPES,
    client_secrets_file="./credentials.json"
)

calendar_service = build_calendar_service(credentials)

createeventtool = CreateGoogleCalendarEvent.from_api_resource(calendar_service)

# %%
# calendar functions
class CalendarResponse(BaseModel):
    event: str = Field(description="Title of event")
    start: str = Field(description="Start time of event")
    end: str = Field(description="End time of event")
    location: Optional[str] = Field(description="Location of event")
    description: Optional[str] = Field(description="Description of event")

def get_valid_datetime():
    while True:
        start_datetime_str = builtins.input()
        try:
            # Use dateparser to parse the input string
            settings = {
                'TIMEZONE': 'America/Chicago',
                'RETURN_AS_TIMEZONE_AWARE': True,
                'DATE_ORDER': 'MDY'
            }
            start_datetime = dateparser.parse(start_datetime_str, settings=settings)
            
            if start_datetime is None:
                raise ValueError("Could not parse the date")

            # Convert to UTC
            start_datetime = start_datetime.astimezone(pytz.UTC)
            
            # If we get here, parsing was successful
            return start_datetime.isoformat()
        except ValueError:
            logger.warning(f"Could not parse '{start_datetime_str}'. Please try again with a clearer date and time.")
            print("Examples of valid formats:")
            print("- Relative: 'now', 'today', 'today 3pm', 'tomorrow 2pm', 'next Monday 9am', 'in 3 days'")
            print("- Exact: '2023-05-15 10:00', 'May 15, 2023 10:00 AM', '15/05/2023 10:00'")

def check_calendar_missing_fields(state: AgentState) -> AgentState:
    logger.debug("Checking for missing calendar fields.")
    if state['calendar'] is None:
        state['calendar'] = CalendarState()

    chat = state['llm_state'].get_llm(is_private=state['contains_private_info'])
    # chat = ChatOllama(model="llama3.2")
    structured_llm = chat.with_structured_output(CalendarFields)

    prompt = f"""
    Parse the following user message for calendar event details. Extract:
    - summary: The event title or name
    - start_datetime: Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
    - end_datetime: End time in ISO format (YYYY-MM-DDTHH:MM:SS)
    - location: Where the event takes place
    - description: Additional details about the event

    If a field is not found in the message, don't try to guess - just return null for that field.
    For dates and times, convert any relative or natural language times to absolute timestamps.
    
    User message: {state['user_message']}
    Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    try:
        parsed_fields = structured_llm.invoke(prompt)
        
        # Update calendar state with parsed fields
        for field in ['summary', 'start_datetime', 'end_datetime', 'location', 'description']:
            value = getattr(parsed_fields, field)
            if value not in [None, '', '<UNKNOWN>']:
                setattr(state['calendar'], field, value)
    except Exception as e:
        logger.error(f"Error parsing calendar fields: {str(e)}")

    # Check for missing required fields
    required_fields = ['summary', 'start_datetime']
    missing_fields = [field for field in required_fields if not getattr(state['calendar'], field)]
    
    # Add end_datetime to missing fields if it's not present
    if not state['calendar'].end_datetime:
        missing_fields.append('end_datetime')

    state['calendar'].missing_fields = missing_fields

    logger.debug(f"Calendar state after checking for missing fields: {state['calendar']}")
    return state

def ask_calendar_missing_fields(state: AgentState) -> AgentState:
    logger.debug("Asking for missing calendar fields.")
    if state['calendar'] is None or not state['calendar'].missing_fields:
        raise ValueError("Calendar state is not properly initialized or no missing fields identified")

    missing_fields = state['calendar'].missing_fields

    if 'summary' in missing_fields:
        sys.stdout.write(f"\n{Colors.YELLOW}Please provide the name of the event:{Colors.END} ")
        sys.stdout.flush()
        state['calendar'].summary = input().strip()

    
    if 'start_datetime' in missing_fields:
        sys.stdout.write(f"\n{Colors.YELLOW}Please provide the start date and time of the event{Colors.END} (e.g. 'tomorrow 3pm', 'next Monday 2pm'):")
        sys.stdout.flush()
        state['calendar'].start_datetime = get_valid_datetime()

    if 'end_datetime' in missing_fields:
        sys.stdout.write(f"\n{Colors.YELLOW}Please provide the end date and time of the event{Colors.END} (e.g. 'tomorrow 4pm', 'next Monday 3pm'):")
        sys.stdout.flush()
        state['calendar'].end_datetime = get_valid_datetime()

    if 'location' in missing_fields:
        state['calendar'].location = builtins.input("Please provide the location of the event (or press Enter if none): ")

    if 'description' in missing_fields:
        state['calendar'].description = builtins.input("Please provide a description of the event (or press Enter if none): ")

    # Clear the missing_fields list
    state['calendar'].missing_fields = []

    logger.debug(f"Updated calendar state after asking for missing fields: {state['calendar']}")
    return state


def draft_calendar_event(state: AgentState) -> AgentState:
    logger.debug("Drafting calendar event.")
    if state['calendar'] is None:
        raise ValueError("Calendar state is not initialized")
    
    llm = state['llm_state'].get_llm(is_private=state['contains_private_info'])
    # llm = ChatOllama(model="llama3.2")

    parser = PydanticOutputParser(pydantic_object=CalendarResponse)
    # Prepare the prompt for ChatOllama
    calendar_prompt_template = """ 
You are creating a calendar event. Use the following information:
Event: {summary}
Start: {start_datetime}
End: {end_datetime}
Location: {location}
Description: {description}

additional_instructions: {changes}

format insructions: {format_instructions}
"""
    prompt = PromptTemplate(
        template=calendar_prompt_template,
        input_variables=["summary", "start_datetime", "end_datetime", "location", "description", "changes"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
            }
    )
    
    calendar_chain = prompt | llm | parser
    try:
        logger.debug("Invoking LLM for drafting calendar event.")
        response = calendar_chain.invoke({
            "summary": state['calendar'].summary,
            "start_datetime": state['calendar'].start_datetime,
            "end_datetime": state['calendar'].end_datetime,
            "location": state['calendar'].location,
            "description": state['calendar'].description,
            "changes": ",".join(state['calendar'].changes_requested)
        })
        state['calendar'].summary = response.event
        state['calendar'].start_datetime = response.start
        state['calendar'].end_datetime = response.end
        state['calendar'].location = response.location
        state['calendar'].description = response.description
        logger.debug("Calendar event drafted successfully.{state['calendar']}")
    except Exception as e:
        state['calendar'].error = f"Error drafting calendar event: {str(e)}"
        state['changes_requested'].append(state['calendar'].error)
    
    return state

def calendar_approval_and_changes(state: AgentState) -> AgentState:
    logger.debug("calendar changes")
    if state['calendar'] is None:
        raise ValueError("Calendar state is not properly initialized")
    sys.stdout.write(f""" \n{Colors.LIGHT_CYAN}
Event: {state['calendar'].summary}
Start: {state['calendar'].start_datetime}
End: {state['calendar'].end_datetime}
Location: {state['calendar'].location}
Description: {state['calendar'].description}
{Colors.END}
""")
    sys.stdout.flush()
    sys.stdout.write(f"\n{Colors.YELLOW}Do you approve this event? (yes/no):{Colors.END} ")
    sys.stdout.flush()
    approval = input().lower().strip()
    state['approved'] = approval == 'yes'

    if not state['approved']:
        sys.stdout.write(f"\n{Colors.YELLOW}Please specify the changes needed or provide additional instructions:{Colors.END} ")
        sys.stdout.flush()
        changes = input().strip()
        if 'changes_requested' not in state['calendar']:
            state['calendar'].changes_requested = []
        state['calendar'].changes_requested.append(changes)
    return state

def create_calendar_event(state: AgentState) -> AgentState:
    """Create a calendar event with proper handling of unknown values and timezone awareness"""
    
    # Clean and validate the calendar data
    def clean_field(value: str) -> Optional[str]:
        """Clean field values, converting <UNKNOWN> to None"""
        if value in ['<UNKNOWN>', '<unknown>', None, '']:
            return None
        return value

    # Default event duration if end time is unknown (1 hour)
    DEFAULT_DURATION = timedelta(hours=1)
    
    try:
        # Clean all fields
        summary = clean_field(state['calendar'].summary)
        start_datetime_str = clean_field(state['calendar'].start_datetime)
        end_datetime_str = clean_field(state['calendar'].end_datetime)
        location = clean_field(state['calendar'].location)
        description = clean_field(state['calendar'].description)
        timezone = state['calendar'].timezone or 'America/Chicago'
        tz = pytz.timezone(timezone)

        # Validate required fields
        if not summary:
            raise ValueError("Event summary/title is required")
        if not start_datetime_str:
            raise ValueError("Start time is required")

        # Parse start time and ensure timezone awareness
        try:
            start_datetime = parser.parse(start_datetime_str)
            if start_datetime.tzinfo is None:
                start_datetime = tz.localize(start_datetime)
            else:
                start_datetime = start_datetime.astimezone(tz)
        except Exception as e:
            raise ValueError(f"Invalid start time format: {e}")

        # Handle end time
        if not end_datetime_str:
            # If no end time, default to start time + 1 hour
            end_datetime = start_datetime + DEFAULT_DURATION
        else:
            try:
                end_datetime = parser.parse(end_datetime_str)
                if end_datetime.tzinfo is None:
                    end_datetime = tz.localize(end_datetime)
                else:
                    end_datetime = end_datetime.astimezone(tz)
            except Exception as e:
                logger.warning(f"Invalid end time format, using default duration: {e}")
                end_datetime = start_datetime + DEFAULT_DURATION

        # Ensure end time is after start time
        if end_datetime <= start_datetime:
            end_datetime = start_datetime + DEFAULT_DURATION
            logger.warning("End time was before or equal to start time, using default duration")

        # Prepare the event input
        tool_input = {
            "start_datetime": start_datetime.isoformat(),
            "end_datetime": end_datetime.isoformat(),
            "summary": summary,
            "timezone": timezone
        }

        # Add optional fields if they exist
        if location:
            tool_input["location"] = location
        if description:
            tool_input["description"] = description

        # Create the event
        event_result = createeventtool.run(tool_input)
        # print(f"\nEvent created successfully: {event_result}")

        # Update state with the created event details
        state['calendar'].end_datetime = end_datetime.isoformat()
        state['calendar'].event_result = event_result
        state['calendar'].error = None  # Clear any previous errors

    except Exception as e:
        error_msg = f"Error creating calendar event: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(error_msg)
        state['changes_requested'].append(error_msg)
        state['calendar'].error = error_msg  # Now this will work with the updated model

    return state

# privacy check and routing

# Function to check for private information
def check_private_info(state: AgentState) -> AgentState:
    logger.debug("Entering check_private_info function")
    message = state['user_message']
    if "<private>" in message.lower() or "<sensitive>" in message.lower() or "<confidential>" in message.lower():
        print(f"{Colors.GREEN}Private data detected. Using local agent.{Colors.END}")
        state['contains_private_info'] = True
        return state
    if "<public>" in message.lower() or "<email>" in message.lower() or "<contact>" in message.lower():
        print(f"{Colors.YELLOW}Public data detected. Using public agent.{Colors.END}")
        state['contains_private_info'] = False
        return state
    
    chat = state['llm_state'].get_llm(is_private=True)
    structured_llm = chat.with_structured_output(PrivacyCheck)
    
    prompt = f"""
    Analyze the following message and determine if it contains any private or sensitive information.
    Private or sensitive information includes but is not limited to:
    - Personal identification numbers (e.g., SSN, driver's license)
    - Financial information (e.g., credit card numbers, bank account details)
    - Medical information
    - Confidential business information
    - Home addresses or phone numbers
    - Passwords or access codes

    Message: {state['user_message']}
    """
    
    try:
        logger.debug("Generating response from ChatOllama")
        result = structured_llm.invoke(prompt)
        state['contains_private_info'] = result.contains_private_info
        logger.debug(f"Privacy check result: {state['contains_private_info']}")
        # We don't store the explanation in the state as it's not part of AgentState
    except Exception as e:
        # Handle any errors that occur during the ChatOllama call
        logger.error(f"Error in privacy check: {str(e)}")
        state['contains_private_info'] = True  # Err on the side of caution
    
    return state

def route_task(state: AgentState) -> AgentState:
    logger.debug("Entering route_task function")
    message = state['user_message']
    
    if "<pdf>" in message.lower() or "<document>" in message.lower(): 
        print(f"{Colors.GREEN}PDF-related query detected. Using PDF agent.{Colors.END}")
        state['task_type'] = TaskType(task="pdf", explanation="User message contains 'PDF' or 'document'")
        return state
    if "<email>" in message.lower() in message.lower():
        print(f"{Colors.GREEN}Public query detected. Using public agent.{Colors.END}")
        state['task_type'] = TaskType(task="email", explanation="User message contains 'public', 'email', or 'contact'")
        return state
    if "<calendar>" in message.lower() or "<event>" in message.lower():
        print(f"{Colors.GREEN}Calendar-related query detected. Using calendar agent.{Colors.END}")
        state['task_type'] = TaskType(task="calendar", explanation="User message contains 'calendar' or 'event'")
        return state
    if "<search>" in message.lower() or "<find>" in message.lower():
        print(f"{Colors.GREEN}Search-related query detected. Using search agent.{Colors.END}")
        state['task_type'] = TaskType(task="search", explanation="User message contains 'search' or 'find'")
        return state
    
    if state['task_type'] is None or state['task_type'].task not in ["email", "calendar", "pdf", "search"]:
        logger.debug("Task type not set or invalid, attempting to infer")
        chat = state['llm_state'].get_llm(is_private=state['contains_private_info'])
        # chat = ChatOllama(model="llama3.2")
        structured_llm = chat.with_structured_output(TaskType)
        
        prompt = f"""
        You are tasked with classifying user messages into one of the following task types:
        - **email**: Tasks related to composing, sending, or managing emails. If the message contains "email" in any sense classify it as "email"
        - **calendar**: Tasks involving scheduling, managing events, or interacting with a calendar. If the message contains event or schedule or calendar in any sense classify it as "pdf"
        - **pdf**: Tasks that involve PDF documents, such as reading, extracting information, or answering questions. If the message contains "pdfs" in any sense classify it as "pdf"
        - **search**: Tasks that require searching for information on the web or elsewhere.

        Analyze the user message carefully, identify key phrases and details, and choose the most appropriate task type from the list. Only return one of these options: "email", "calendar", "pdf", or "search".

        User message: {state['user_message']}
        """
        
        try:
            logger.debug("Generating response from ChatOllama")
            result = structured_llm.invoke(prompt)
            # print(Colors.GREEN,result, Colors.END)
            # stream_print(f"Task type inferred: {result.task} explanation: {result.explanation}\n", color=Colors.GREEN)
            state['task_type'] = result
        except Exception as e:
            logger.error(f"Error in task routing: {str(e)}")
            if isinstance(result, dict) and 'task' in result:
                try:
                    normalized_result = {
                        'task': result['task'].lower(),
                        'explanation': result['explanation']
                    }
                    state['task_type'] = TaskType(**normalized_result)
                    return state
                except Exception as inner_e:
                    logger.error(f"Error in normalization attempt: {str(inner_e)}")
    
    return state



# %%

workflow = StateGraph(AgentState)
workflow.add_node("check_private", check_private_info)
workflow.add_node("route_task", route_task)

# Email nodes
workflow.add_node("check_email_fields", check_email_missing_fields)
workflow.add_node("ask_email_missing_fields", ask_email_missing_fields)
workflow.add_node("draft_email", draft_email)
workflow.add_node("user_approval_and_changes", user_approval_and_changes)
workflow.add_node("send_email", send_email_node)

# Calendar nodes
workflow.add_node("check_calendar_fields", check_calendar_missing_fields)
workflow.add_node("ask_calendar_missing_fields", ask_calendar_missing_fields)
workflow.add_node("calendar_approval_and_changes", calendar_approval_and_changes)
workflow.add_node("draft_event", draft_calendar_event)
workflow.add_node("create_event", create_calendar_event)

# PDF nodes
workflow = add_pdf_nodes(workflow)

# Search nodes
# workflow.add_node("internet_search", internet_search)
workflow = add_search_nodes(workflow)

# Set entry point
workflow.set_entry_point("check_private")

# Add edges for initial private info check
workflow.add_conditional_edges(
    "check_private",
    lambda x: "local_llm" if x['contains_private_info'] else "public_llm",
    {
        "local_llm": "route_task",
        "public_llm": "route_task"
    }
)

# Add edges for task routing
workflow.add_conditional_edges(
    "route_task",
    lambda x: x['task_type'].task,
    {
        "email": "check_email_fields",
        "calendar": "check_calendar_fields",
        "pdf": "validate_pdf",
        "search": "internet_search"
    }
)

# Email workflow edges
workflow.add_conditional_edges(
    "check_email_fields",
    lambda x: "ask_missing_fields" if x['email'].get('missing_fields') else "draft_email",
    {
        "ask_missing_fields": "ask_email_missing_fields",
        "draft_email": "draft_email"
    }
)

workflow.add_edge("ask_email_missing_fields", "draft_email")
workflow.add_edge("draft_email", "user_approval_and_changes")
workflow.add_conditional_edges(
    "user_approval_and_changes",
    lambda x: "send_email" if x['approved'] else "draft_email",
    {
        "send_email": "send_email",
        "draft_email": "draft_email"
    }
)
workflow.add_edge("send_email", END)

# Calendar workflow edges
workflow.add_conditional_edges(
    "check_calendar_fields",
    lambda x: "ask_missing_fields" if x['calendar'].missing_fields else "calendar_approval_and_changes",
    {
        "ask_missing_fields": "ask_calendar_missing_fields",
        "calendar_approval_and_changes": "calendar_approval_and_changes"
    }
)
# workflow.add_edge("ask_calendar_missing_fields", "create_event")
workflow.add_edge("ask_calendar_missing_fields", "draft_event")
workflow.add_edge("draft_event", "calendar_approval_and_changes")
workflow.add_conditional_edges(
    "calendar_approval_and_changes",
    lambda x: "create" if x['approved'] else "calendar_approval_and_changes",
    {
        "create": "create_event",
        "calendar_approval_and_changes": "draft_event"
    }
)
workflow.add_edge("create_event", END)

# Compile the workflow
app = workflow.compile()




# %%


def is_positive_feedback(text: str) -> bool:
    """Check if the input is positive feedback"""
    positive_phrases = [
        'good', 'great', 'awesome', 'nice', 'thanks', 'thank you',
        'well done', 'excellent', 'amazing', 'perfect', 'good work',
        'good job', 'wonderful', 'fantastic', 'brilliant', 'cool'
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in positive_phrases)

def get_random_response() -> str:
    """Get a random friendly response"""
    responses = [
        "I'm glad I could help! What else would you like to know?",
        "Thank you! I'm here to help with anything else you need.",
        "That's kind of you! Let me know if you need anything else.",
        "I appreciate your feedback! What else can I assist you with?",
        "Happy to help! Feel free to ask me anything else.",
        "Thank you for the kind words! What's next on your mind?",
        "I'm delighted to be helpful! What other questions do you have?",
        "That means a lot! How else can I assist you today?",
        "You're welcome! Ready for your next question!",
        "Thank you! Looking forward to helping you more!"
    ]
    return f"{Colors.LIGHT_GREEN}{random.choice(responses)}{Colors.END}"

def format_step_output(step: dict) -> str:
    """Format the step output as colored text"""
    step_name = list(step.keys())[0]
    if step_name == 'END':
        return f"\n{Colors.LIGHT_GREEN}Workflow completed.{Colors.END}\n"
    
    state = step[step_name]
    output = f"{Colors.LIGHT_PURPLE}Processing step {step_name}... {Colors.END}\n"
    if step_name == 'check_calendar_fields' or step_name == 'ask_calendar_missing_fields' or step_name == "calendar_approval_and_changes":
        return output
    
    if step_name == "route_task":
        output += f"""{Colors.LIGHT_CYAN}Task Type Inferred:{Colors.END} {state['task_type'].task}\n
        {Colors.LIGHT_CYAN}Explanation:{Colors.END} {state['task_type'].explanation}\n"""
        return output

    if state['task_type'] and state['task_type'].task == 'email':
        if step_name in ['draft_email'] and 'email' in state and state['email']:
            if 'subject' in state['email'] and 'body' in state['email']:
                output += f"""
{Colors.LIGHT_CYAN}Draft Email:{Colors.END}
{Colors.YELLOW}Subject:{Colors.END} {state['email']['subject']}

{Colors.YELLOW}Body:{Colors.END}
{state['email']['body']}\n"""
            
        if step_name == 'send_email' and 'email' in state and state['email']:
            output += f"\n{Colors.LIGHT_GREEN}Email Sent Successfully{Colors.END}\n"
    
    elif state['task_type'] and state['task_type'].task == 'calendar':
        if 'calendar' in state and state['calendar'] and state['calendar'].error is None and state['calendar'].event_result is None:
#             output = f"""
# {Colors.LIGHT_CYAN}Calendar Event:{Colors.END}
# {state['calendar'].event_result}\n"""
            output += f""" {Colors.LIGHT_CYAN}
Event: {state['calendar'].summary}
Start: {state['calendar'].start_datetime}
End: {state['calendar'].end_datetime}
Location: {state['calendar'].location}
Description: {state['calendar'].description}
{Colors.END}
"""
        elif 'calendar' in state and state['calendar'] and state['calendar'].event_result:
            output += f"{Colors.LIGHT_GREEN}Created event: {state['calendar'].event_result}{Colors.END}"
        
        elif state['calendar'] and state['calendar'].error:
            output += f"{Colors.RED}Error occured {state['calendar'].error}{Colors.END}"
        stream_print(output)
        return
    elif state['task_type'] and state['task_type'].task == 'search':
        if 'search' in state and state['search'] and state['search'].get('final_response'):
            output += f"""
{Colors.LIGHT_CYAN}Search Results:{Colors.END}
{state['search']['final_response']}\n"""

    smart_output(output, color=Colors.LIGHT_CYAN)

# %%



def create_prompt_session():
    """Create a prompt session with placeholder and proper scrolling"""
    # Get terminal size
    terminal_width, terminal_height = shutil.get_terminal_size()
    
    # Define style
    style = Style.from_dict({
        'prompt': '#00aa00 bold',         # Green prompt
        'placeholder': '#666666 italic',   # Gray italic placeholder
        'bottom-toolbar': 'bg:#222222 #aaaaaa',  # Toolbar style
        'output-field': 'bg:#000000 #ffffff',    # Output area style
    })

    # Create custom output
    output = create_output(stdout=None)
    
    # Create session with scrolling support
    session = PromptSession(
        message=HTML('<prompt>You: </prompt>'),
        style=style,
        placeholder='type your message or "help" for usage guide...',
        enable_history_search=True,
        mouse_support=True,
        wrap_lines=True,
        complete_while_typing=True
    )
    return session


def show_help():
    """Display help information with colors"""
    help_text = f"""
{Colors.LIGHT_CYAN}🔍 Help Guide
================================{Colors.END}

This assistant supports both natural language queries and command tags.
Natural language queries may take longer to process as they go through our LLM.
Using command tags can help speed up query processing.

{Colors.YELLOW}Available Tags:{Colors.END}
-----------
1. Email Tasks: {Colors.LIGHT_PURPLE}<email>{Colors.END} 
2. Calendar Tasks: {Colors.LIGHT_PURPLE}<calendar>{Colors.END} 
3. PDF Tasks: {Colors.LIGHT_PURPLE}<pdf>{Colors.END}
4. Search Tasks: {Colors.LIGHT_PURPLE}<search>{Colors.END}

{Colors.YELLOW}Special Tags:{Colors.END}
------------
- {Colors.LIGHT_PURPLE}<private>{Colors.END} : Add this tag when handling sensitive information
- {Colors.LIGHT_PURPLE}<public>{Colors.END} : Add this tag for non-sensitive content

{Colors.YELLOW}Example Tasks:{Colors.END}
------------
{Colors.LIGHT_CYAN}Natural Language:{Colors.END}
- "Send an email to my professor asking for a meeting"
- "Search the internet for the latest AI research papers"
- "Add a reminder for my dentist appointment next week"
- "Tell me about Frontier in the PDF document"

{Colors.LIGHT_CYAN}With Tags (Faster Processing):{Colors.END}
- {Colors.LIGHT_PURPLE}<email>{Colors.END} Draft a meeting request to Professor Smith
- {Colors.LIGHT_PURPLE}<search>{Colors.END} Latest developments in AI 2024
- {Colors.LIGHT_PURPLE}<calendar>{Colors.END} Schedule team meeting Monday 10 AM
- {Colors.LIGHT_PURPLE}<pdf>{Colors.END} Analyze the annual report

{Colors.YELLOW}Additional Commands:{Colors.END}
-----------------
- {Colors.LIGHT_GREEN}help{Colors.END} : Display this help menu
- {Colors.LIGHT_GREEN}quit{Colors.END} : Exit the application

{Colors.YELLOW}Tips:{Colors.END}
----
• Use Page Up/Down to scroll
• Press 'q' to exit this help
• Use Ctrl+C to cancel operations
"""
    with console.pager():
        console.print(Panel(help_text, title="Help Guide", border_style="cyan"))

def run_interactive_agent():
    """Run the agent with improved console output"""
    current_state = {
        "task_type": None,
        "contains_private_info": False,
        "user_message": "",
        "email": None,
        "calendar": None,
        "pdf": None,
        "search": None,
        "approved": False,
        "changes_requested": [],
        "llm_state": LLMState()
    }

    welcome_message = f"""
{Colors.LIGHT_CYAN}Welcome to EDITH! 🤖{Colors.END}
I am your personal assistant and can help you with various tasks:
{Colors.YELLOW}• Sending emails 📧
• Scheduling events 📅
• Extracting information from PDFs 📄
• Web searches 🔍{Colors.END}

Type {Colors.LIGHT_GREEN}'help'{Colors.END} for usage guide or start with a command (e.g., {Colors.LIGHT_PURPLE}<email>{Colors.END}, {Colors.LIGHT_PURPLE}<calendar>{Colors.END}, {Colors.LIGHT_PURPLE}<search>{Colors.END})
"""
    stream_print(welcome_message)
    
    while True:
        try:
            # Simple input prompt
            sys.stdout.write(f"\n{Colors.CYAN}You: {Colors.END}")
            sys.stdout.flush()
            user_input = input().strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                stream_print(f"\n{Colors.GREEN}Goodbye! Have a great day! {Colors.END}👋\n")
                break
                
            if user_input.lower() == 'help':
                show_help()
                continue

            if is_positive_feedback(user_input):
                response = get_random_response()
                stream_print(f"\n{response}\n")
                continue
            
            if not user_input:
                stream_print("\nPlease enter a valid input.\n")
                continue

            # Update state
            current_state.update({
                "user_message": user_input,
                "task_type": None,
                "approved": False,
                "changes_requested": []
            })

            # Reset task states
            for task_type in ['email', 'calendar', 'search']:
                if f"<{task_type}>" in user_input.lower():
                    current_state[task_type] = None

            stream_print("\nProcessing your request...\n", color=Colors.PURPLE)
            
            # Process steps
            last_step_name = None
            for step in app.stream(current_state):
                step_name = list(step.keys())[0]
                if step_name != last_step_name:
                    # output = format_step_output(step)
                    # if output:
                    #     stream_print(output)
                    format_step_output(step)
                    last_step_name = step_name

        except KeyboardInterrupt:
            stream_print("\nOperation cancelled.\n", color=Colors.LIGHT_RED)
            continue
        except Exception as e:
            logger.debug(f"Error in processing: {str(e)}")
            stream_print(f"\nError: {str(e)}\n")
            continue

def stream_user_input(prompt: str = "") -> str:
    """Stream the user input prompt and get input.
    
    Args:
        prompt (str): The prompt to display
        
    Returns:
        str: User input
    """
    stream_print(prompt, delay=0.02)
    return input()


run_interactive_agent()

# current_state = {
#         "task_type": None,
#         "contains_private_info": False,
#         "user_message": "",
#         "email": {
#             "missing_fields": [],
#             "subject": "hello",
#             "body": "world",
#             "to": "2k2vamshi@gmail.com",
#         },
#         "calendar": None,
#         "pdf": None,
#         "search": None,
#         "approved": False,
#         "changes_requested": [],
#         "llm_state": LLMState()
#     }

# draft_email(current_state)