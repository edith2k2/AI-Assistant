# AI Assistant

## Overview

This project implements an AI assistant that performs various tasks based on user input. The assistant intelligently selects between local and public Large Language Models (LLMs) depending on the nature of the prompt:

- **Local LLM**: Utilized for sensitive or private tasks using Llama 3.2 via Ollama.
- **Public LLM**: Accessed for general, non-sensitive information or tasks that require external data, using Claude 3.5 Sonnet.

The assistant supports sending emails, searching the internet, and managing calendar events using LangChain and Composio for efficient task execution.

## Features

- **Dynamic Model Selection**: Automatically determines whether to use the local LLM (via Llama 3.2 via Ollama) or a public LLM (Claude 3.5 Sonnet) based on the content of the prompt.
- **Email Automation**: Capable of drafting and sending emails as per user instructions.
- **Internet Search**: Searches the web for information in response to user queries.
- **Calendar Management**: Adds and manages events using LangChain and Composio for calendar integration and synchronization.

## Requirements

- **LangChain**: For interacting with LLMs and managing prompts.
- **Llama 3.2 via Ollama**: Hosts the local LLM.
- **Python Packages**:
    - `requests`
    - `langchain`

To install dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the HW3.py**:
     - Run the HW3.py. An interactive console opens in terminal.
2. **Interact with the Assistant**:
     - The assistant will ask for a prompt. Based on the content:
         - If the task involves private or sensitive data, the assistant uses the Local LLM (hosted via Llama 3.2 via Ollama).
         - For tasks involving public information (e.g., web search), the assistant uses the Public LLM (Claude 3.5 Sonnet).
3. **Example Tasks**:
     - Email: “Send an email to my professor asking for a meeting.”
     - Internet Search: “Search the internet for the latest AI research papers.”
     - Calendar: “Add a reminder to submit my assignment next Monday.”

### Example Tasks:

#### Natural Language:
- "Send an email to my professor asking for a meeting"
- "Search the internet for the latest AI research papers"
- "Add a reminder for my dentist appointment next week"
- "Tell me about Frontier in the PDF document"

#### With Tags (Faster Processing):
- `<email>` Draft a meeting request to Professor Smith for tomorrow
- `<search>` Latest developments in artificial intelligence 2024
- `<calendar>` Schedule team meeting every Monday at 10 AM
- Tell me his work at NVIDIA `<pdf>`
- `<email>` Send salary details to HR `<private>`
- `<calendar>` `<public>` Add company holiday party on December 20th

## Configuration

- **Local LLM (Llama 3.2 via Ollama)**: Ensure that Llama 3.2 via Ollama is set up and running on your system. Update any paths or settings in the notebook to match your local environment.
- **API Keys**: For public LLMs, internet search, and calendar integrations, set your API keys in the notebook configuration section:

```python
ANTHROPIC_API_KEY = "your_public_llm_api_key"
```
- **Google Authentication**: Ensure that `credentials.json` is present in the folder.
  - To use Google services (Calendar and Gmail), you'll need to set up Google authentication:

  1. Visit [Google Cloud Console](https://console.cloud.google.com/)
  2. Create a new project or select an existing project
  3. Navigate to the "Credentials" section
  4. Click "Create Credentials" and select "OAuth 2.0 Client IDs"
  5. Configure the consent screen
  6. Download the `credentials.json` file
  7. Place the `credentials.json` file in your project directory

## Future Enhancements

### Planned Features
- **Directory Structure Optimization**
 - Implement organized folder hierarchy for better code management
 - Separate modules by functionality (auth, services, utils)
 - Create dedicated folders for configuration files
 - Add template folders for user customization

## License

This project is licensed under the MIT License.

## Contact

- **Email**: blankmask@gmail.com
- **GitHub**: [https://github.com/edith2k2/AI-Assistant](https://github.com/edith2k2/AI-Assistant)
