# Agentify Example: RLM

Example code for agentifying RLM using A2A and MCP standards.

Some code in this repo is taken from https://github.com/alexzhang13/rlm-minimal/

## Project Structure

```
src/
├── green_agent/    # Assessment manager agent
├── white_agent/    # Target agent being tested
└── launcher.py     # Evaluation coordinator
```

## Installation

```bash
uv sync
```

## Usage

First, configure `.env` with `OPENAI_API_KEY=...`, then

```bash
# Launch complete evaluation
uv run python main.py launch
```
