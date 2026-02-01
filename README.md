# Agentify RLM

Agentifying [RLM](https://arxiv.org/abs/2512.24601) using A2A and MCP standards.

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

```bash
# Launch green agent only
uv run python main.py green
```

```bash
# Build and push docker containers
./build-and-push.sh
```
