# Multi-Provider LLM Client

A reusable async Python module abstracting OpenAI, Anthropic, and Gemini behind a common interface, with intelligent routing, automatic fallback, and persistent cost tracking.

## Features

- Unified `generate()` async interface across all 3 providers
- Streaming support (`stream=True`) returning a custom `AsyncIterator[str]`
- Structured output via Pydantic schemas (`schema=MyModel`)
- Smart router: hard filters (context window, exclusions) + soft sort (`cheapest` / `fastest` / `most_capable`)
- Automatic fallback: 429 → immediate; timeout → 1 retry then fallback; any other error → immediate
- Cost tracker: in-memory + auto-flush to `.jsonl`

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
cd 04_multi_provider
uv sync
```

Create a `.env` file at the **repo root** (one level above `04_multi_provider/`):

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
```

## Usage

### Basic generation

```python
from llm_client import OpenAIProvider, AnthropicProvider, GeminiProvider

openai = OpenAIProvider()
response = await openai.generate(
    prompt="Explain what a neural network is in 2 sentences.",
    model="gpt-5.4",
)
print(response.text)
print(f"Cost: ${response.cost_usd:.6f} | Latency: {response.latency_ms}ms")
```

### Structured output

```python
from pydantic import BaseModel
from llm_client import AnthropicProvider

class Summary(BaseModel):
    title: str
    body: str

anthropic = AnthropicProvider()
response = await anthropic.generate(
    prompt="Summarise transformers in one sentence.",
    model="claude-sonnet-4-6",
    schema=Summary,
)
print(response.parsed.title)   # validated Pydantic object
```

### Streaming

```python
from llm_client import OpenAIProvider, StreamingResponse

openai = OpenAIProvider()
result = await openai.generate(prompt="Tell me a joke.", model="gpt-5.4", stream=True)

async for chunk in result:
    print(chunk, end="", flush=True)

final = result.final_response   # available after iterator is exhausted
```

### Routing and fallback

```python
from llm_client import OpenAIProvider, AnthropicProvider, generate_with_fallback

providers = {"openai": OpenAIProvider(), "anthropic": AnthropicProvider()}
models = ["gpt-5.4", "claude-sonnet-4-6"]

response = await generate_with_fallback(
    prompt="Hello!",
    providers=providers,
    available_models=models,
    strategy="cheapest",   # "cheapest" | "fastest" | "most_capable"
)
```

### Cost tracking

```python
from pathlib import Path
from llm_client import OpenAIProvider, CostTracker

tracker = CostTracker(log_path=Path("costs.jsonl"))
openai = OpenAIProvider(cost_tracker=tracker)

await openai.generate(prompt="Hello", model="gpt-5.4")
print(f"Total cost so far: ${tracker.total_cost():.6f}")
```

## Supported models

| Model | Provider | Context window | Tier |
|---|---|---|---|
| `gpt-5.4` | openai | 1 050 000 | 3 |
| `claude-opus-4-6` | anthropic | 1 000 000 | 3 |
| `claude-sonnet-4-6` | anthropic | 1 000 000 | 2 |
| `gemini-3.1-pro` | gemini | 1 000 000 | 3 |

## Running tests

```bash
# Unit tests only (no API keys required)
uv run pytest tests/unit/ -v

# Integration tests (requires real API keys in ../../.env)
uv run pytest tests/integration/ -v -m integration
```

## Demo notebook

Open `demo.ipynb` in JupyterLab for an interactive walkthrough:

```bash
uv run --extra notebooks jupyter lab demo.ipynb
```
