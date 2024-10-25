# AI Assistant

## Overview

This Jupyter Notebook project implements an AI assistant that performs various tasks based on user input. The assistant intelligently selects between local and public Large Language Models (LLMs) depending on the nature of the prompt:

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
- **Composio**: For handling calendar events.
- **Llama 3.2 via Ollama**: Hosts the local LLM.
- **Python Packages**:
    - `requests`
    - `langchain`
    - `composio`

To install dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

1. **Launch the Notebook**:
     - Open the notebook file (`AI_Assistant.ipynb`) using Jupyter Notebook or Jupyter Lab.
2. **Interact with the Assistant**:
     - The assistant will ask for a prompt. Based on the content:
         - If the task involves private or sensitive data, the assistant uses the Local LLM (hosted via Llama 3.2 via Ollama).
         - For tasks involving public information (e.g., web search), the assistant uses the Public LLM (Claude 3.5 Sonnet).
3. **Example Tasks**:
     - Email: “Send an email to my professor asking for a meeting.”
     - Internet Search: “Search the internet for the latest AI research papers.”
     - Calendar: “Add a reminder to submit my assignment next Monday.”

## Configuration

- **Local LLM (Llama 3.2 via Ollama)**: Ensure that Llama 3.2 via Ollama is set up and running on your system. Update any paths or settings in the notebook to match your local environment.
- **API Keys**: For public LLMs, internet search, and calendar integrations, set your API keys in the notebook configuration section:

```python
PUBLIC_LLM_API_KEY = "your_public_llm_api_key"
```

## Future Enhancements

- Enhancing calendar functionalities for more detailed event management.
- Improving email templates and scheduling capabilities.
- Adding PDF QA answering capabilites

## License

This project is licensed under the MIT License.

## Contact

- **Email**: blankmask@gmail.com
- **GitHub**: [https://github.com/edith2k2/AI-Assistant](https://github.com/edith2k2/AI-Assistant)
