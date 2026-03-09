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

## Quick Start
### Command Line

# Run a single task
reflexion "What is the capital of France?"

# Run with specific environment
reflexion --env production "Explain quantum computing"

# Interactive mode
reflexion --interactive

# Verbose output
reflexion --verbose "Calculate 15 * 27"

# JSON output
reflexion --format json "Who wrote Romeo and Juliet?"

### Configuration
Create a .env file:
```bash
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
ENV=development
```

### Architecture
┌─────────────────────────────────────────────────────┐
│                    Reflexion Agent                  │
├───────────────┬─────────────────┬───────────────────┤
│     Actor      │    Evaluator     │    Reflector    │
│  (Generates    │   (Scores and    │   (Learns from  │
│   answers)     │    critiques)    │    failures)    │
├───────────────┴─────────────────┴───────────────────┤
│                    Memory System                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │  Episodic  │  │ Reflection │  │Short-term  │     │
│  │   Store    │  │   Store    │  │  Memory    │     │
│  └────────────┘  └────────────┘  └────────────┘     │
├─────────────────────────────────────────────────────┤
│                    Tools System                     │
│       Calculator    Web Search    Extensible        │
└─────────────────────────────────────────────────────┘

### Project Structure

reflexion-agent/
├── src/
│   └── reflexion_agent/
│       ├── agent/          # Actor, Evaluator, Reflector, Loop
│       ├── config/         # Settings and configuration
│       ├── memory/         # Episodic, reflection, short-term
│       ├── observability/  # Logging, tracing, metrics
│       ├── prompts/        # Prompt templates
│       ├── providers/      # LLM providers (OpenAI, Anthropic)
│       ├── tools/          # Calculator, web search, registry
│       └── utils/          # Exceptions, helpers
├── configs/                # YAML configuration files
├── examples/               # Usage examples
├── tests/                  # Unit and integration tests
├── .env.example            # Example environment variables
├── pyproject.toml          # Project configuration
└── README.md               # This file
