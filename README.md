# Reflexion Agent

A AI agent with episodic and reflection memory that learns from its mistakes through a sophisticated reflexion loop.

## Features

-  **Reflexion Loop**: Act → Evaluate → Reflect → Improve
-  **Episodic Memory**: Long-term storage of past experiences
-  **Reflection Memory**: Learn from failures and successes
-  **Tool Integration**: Calculator, web search, and extensible tool framework
-  **Observability**: Structured logging, distributed tracing, metrics
-  **Production Ready**: Configuration management, error handling, retries
-  **Multiple LLMs**: Support for OpenAI and Anthropic (extensible)
-  **Easy Deployment**: CLI, API server, Docker support

## Installation

```bash
# Clone the repository
git clone https://github.com/msaShahid/ReflexionAgent.git
cd reflexion-agent

# Install with pip
pip install -e .

# Or with poetry
poetry install

# Install with development dependencies
pip install -e ".[dev]"
```
