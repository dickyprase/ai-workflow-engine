# AI Workflow Engine

[![CI](https://github.com/dickyprase/ai-workflow-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/dickyprase/ai-workflow-engine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An intelligent automation framework that orchestrates AI-powered workflows for developers.** Supports multi-model routing, task decomposition, and automated pipeline execution.

---

## Features

- **Multi-Model Routing** — Dynamically route tasks to the optimal LLM provider (OpenAI, Anthropic, Google, local models) based on cost, latency, and capability requirements.
- **Intelligent Task Decomposition** — Automatically break complex objectives into atomic, parallelizable subtasks with dependency resolution.
- **Pipeline Orchestration** — Define DAG-based workflows with conditional branching, retry logic, and streaming intermediate results.
- **Async-First Architecture** — Built on `asyncio` for high-throughput concurrent execution with backpressure handling.
- **Configuration as Code** — YAML/JSON pipeline definitions with environment-aware variable interpolation.
- **Extensible Plugin System** — Register custom task handlers, model adapters, and output processors.
- **Built-in Observability** — Structured logging, execution tracing, and cost tracking per workflow run.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Workflow Engine                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────┐    ┌──────────────┐    ┌───────────────────┐     │
│  │  Pipeline  │───▶│    Task      │───▶│   Model Router    │     │
│  │ Orchestrator│   │ Decomposer   │    │                   │     │
│  └───────────┘    └──────────────┘    │  ┌─────┐ ┌─────┐ │     │
│        │                               │  │OpenAI│ │Anth.│ │     │
│        ▼                               │  └─────┘ └─────┘ │     │
│  ┌───────────┐    ┌──────────────┐    │  ┌─────┐ ┌─────┐ │     │
│  │  Scheduler │───▶│   Executor   │───▶│  │Gemini│ │Local│ │     │
│  │  (DAG)    │    │   (Async)    │    │  └─────┘ └─────┘ │     │
│  └───────────┘    └──────────────┘    └───────────────────┘     │
│        │                                        │                 │
│        ▼                                        ▼                 │
│  ┌───────────┐                        ┌───────────────────┐     │
│  │  Config   │                        │  Result Aggregator │     │
│  │  Manager  │                        │  & Cost Tracker    │     │
│  └───────────┘                        └───────────────────┘     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install ai-workflow-engine
```

Or install from source:

```bash
git clone https://github.com/dickyprase/ai-workflow-engine.git
cd ai-workflow-engine
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from engine import WorkflowEngine, PipelineConfig

async def main():
    # Initialize the engine
    engine = WorkflowEngine.from_config("workflow.yaml")

    # Define a simple pipeline
    pipeline = engine.create_pipeline("code-review")
    pipeline.add_task("analyze", model="gpt-4o", prompt="Analyze this code for bugs: {input}")
    pipeline.add_task("suggest", model="claude-sonnet", prompt="Suggest fixes: {analyze.output}")
    pipeline.add_task("summarize", depends_on=["analyze", "suggest"])

    # Execute with input
    result = await engine.run(pipeline, input={"code": "def foo(): ..."})
    print(result.output)
    print(f"Total cost: ${result.metrics.total_cost:.4f}")

asyncio.run(main())
```

## Configuration

Create a `workflow.yaml` file:

```yaml
engine:
  max_concurrency: 10
  default_timeout: 30
  retry_policy:
    max_retries: 3
    backoff_factor: 2.0

models:
  openai:
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4o
    max_tokens: 4096
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-sonnet-4-20250514

routing:
  strategy: cost_optimized  # or: latency_optimized, capability_match
  fallback_chain: [openai, anthropic, local]
```

## API Reference

### WorkflowEngine

| Method | Description |
|--------|-------------|
| `from_config(path)` | Create engine from YAML/JSON config |
| `create_pipeline(name)` | Create a new pipeline instance |
| `run(pipeline, input)` | Execute a pipeline with given input |
| `run_batch(pipeline, inputs)` | Execute pipeline across multiple inputs |

### Pipeline

| Method | Description |
|--------|-------------|
| `add_task(name, **kwargs)` | Add a task node to the pipeline |
| `add_condition(predicate, branches)` | Add conditional branching |
| `set_error_handler(handler)` | Register error handling callback |

### ModelRouter

| Method | Description |
|--------|-------------|
| `route(task)` | Select optimal model for a task |
| `register_provider(name, adapter)` | Add a custom model provider |
| `get_cost_estimate(task)` | Estimate execution cost |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/
black --check src/

# Type checking
mypy src/
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built with purpose by [dickyprase](https://github.com/dickyprase)
