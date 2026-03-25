# MCP Chat

MCP Chat is a command-line interface application that enables interactive chat capabilities with AI models through the Anthropic API. The application supports document retrieval, command-based prompts, and extensible tool integrations via the MCP (Model Control Protocol) architecture.

## Prerequisites

- Python 3.10+
- Anthropic API Key

## Setup

### Step 1: Configure the environment variables

1. Create or edit the `.env` file in the project root and verify that the following variables are set correctly:

```text
ANTHROPIC_API_KEY=""  # Enter your Anthropic API secret key
```

### Step 2: Install dependencies and run

#### Option 1: Setup with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

1. Install uv, if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Install dependencies and run:

```bash
uv sync
uv run main.py
```

`uv sync` creates the virtual environment and installs all dependencies from `uv.lock` in one step.

#### Option 2: Setup without uv

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

1. Install dependencies:

```bash
pip install -e .
```

1. Run the project:

```bash
python main.py
```

## Usage

### Basic Interaction

Simply type your message and press Enter to chat with the model.

### Document Retrieval

Use the @ symbol followed by a document ID to include document content in your query:

```console
> Tell me about @deposition.md
```

### Commands

Use the / prefix to execute commands defined in the MCP server:

```console
> /summarize deposition.md
```

Commands will auto-complete when you press Tab.

### Linting and Typing Check

There are no lint or type checks implemented.
