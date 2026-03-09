# Reflexion Agent
An intelligent AI agent that learns from its mistakes through a sophisticated reflexion loop, featuring episodic and reflection memory. Built for production use with robust observability, multiple LLM support, and extensible tool integration.

## 🚀 Features

- **🔄 Reflexion Loop**: Act → Evaluate → Reflect → Improve - continuous learning from interactions
- **🧠 Episodic Memory**: Long-term storage of past experiences and conversations
- **💭 Reflection Memory**: Learn from failures and successes to improve future performance
- **🛠️ Tool Integration**: Built-in calculator, web search, and extensible tool framework
- **📊 Observability**: Structured logging, distributed tracing, and performance metrics
- **🔧 Production Ready**: Configuration management, error handling, retries, and graceful degradation
- **🤖 Multiple LLMs**: Support for OpenAI (GPT-4, GPT-3.5) and Anthropic (Claude) - easily extensible
- **🚀 Easy Deployment**: CLI, API server, and Docker support

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- API keys for your chosen LLM provider(s)

### Option 1: Install from source
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

### Option 2: Using Docker
```bash
docker build -t reflexion-agent .
docker run -e OPENAI_API_KEY=your-key reflexion-agent
```

## 🏃 Quick Start

### Command Line Interface

```bash
# Run a single task
reflexion "What is the capital of France?"

# Run with specific environment
reflexion --env production "Explain quantum computing"

# Interactive mode for ongoing conversations
reflexion --interactive

# Verbose output for debugging
reflexion --verbose "Calculate 15 * 27"

# JSON output for programmatic use
reflexion --format json "Who wrote Romeo and Juliet?"
```

### Python API

```python
from reflexion_agent import ReflexionAgent

# Initialize the agent
agent = ReflexionAgent(
    llm_provider="openai",  # or "anthropic"
    model="gpt-4",
    config_path="configs/production.yaml"
)

# Run a task
result = agent.run("What is the capital of France?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")

# Interactive session
agent.interactive()
```

### API Server

```bash
# Start the API server
reflexion-server --host 0.0.0.0 --port 8000

# In another terminal
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the capital of France?"}'
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# LLM Providers
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-your-key

# Environment
ENV=development  # or production, testing

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Memory Settings
EPISODIC_MEMORY_SIZE=1000
REFLECTION_MEMORY_SIZE=100
```

### Configuration Files
The `configs/` directory contains YAML configuration files for different environments:

- `configs/development.yaml` - Development settings
- `configs/production.yaml` - Production-optimized settings
- `configs/testing.yaml` - Testing configuration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Reflexion Agent                  │
├───────────────┬─────────────────┬───────────────────┤
│     Actor      │    Evaluator     │    Reflector     │
│  (Generates    │   (Scores and    │   (Learns from   │
│   answers)     │    critiques)    │    failures)     │
├───────────────┴─────────────────┴───────────────────┤
│                    Memory System                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  Episodic  │  │ Reflection │  │Short-term  │    │
│  │   Store    │  │   Store    │  │  Memory    │    │
│  └────────────┘  └────────────┘  └────────────┘    │
├─────────────────────────────────────────────────────┤
│                    Tools System                      │
│       ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│       │Calculator│  │Web Search│  │Extensible│    │
│       └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
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
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Inspired by the [Reflexion](https://arxiv.org/abs/2303.11366) paper by Shinn et al.
- Built with modern LLM best practices
- Community contributions and feedback

## 📧 Contact

Project Link: [https://github.com/msaShahid/ReflexionAgent](https://github.com/msaShahid/ReflexionAgent)

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

---

**Made with ❤️ by [msaShahid](https://github.com/msaShahid)**
```