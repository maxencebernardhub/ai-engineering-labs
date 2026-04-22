# Multi-Provider LLM Client

Async Python module abstracting OpenAI, Anthropic, and Gemini behind a common interface,
with intelligent routing, automatic fallback, and persistent cost tracking.

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

| Model | Provider | Context window | Tier | Input $/MTok | Output $/MTok |
| --- | --- | --- | --- | --- | --- |
| `gpt-5.4` | openai | 1 050 000 | 3 | 2.50 | 15.00 |
| `gpt-5.4-mini` | openai | 400 000 | 2 | 0.75 | 4.50 |
| `gpt-5.4-nano` | openai | 400 000 | 1 | 0.20 | 1.25 |
| `claude-opus-4-7` | anthropic | 1 000 000 | 3 | 5.00 | 25.00 |
| `claude-opus-4-6` | anthropic | 1 000 000 | 3 | 5.00 | 25.00 |
| `claude-sonnet-4-6` | anthropic | 1 000 000 | 2 | 3.00 | 15.00 |
| `claude-sonnet-4-5` | anthropic | 200 000 | 2 | 3.00 | 15.00 |
| `claude-haiku-4-5` | anthropic | 200 000 | 1 | 1.00 | 5.00 |
| `gemini-3.1-pro` | gemini | 1 000 000 | 3 | 2.00 | 12.00 |
| `gemini-2.5-pro` | gemini | 1 000 000 | 3 | 1.25 | 10.00 |
| `gemini-3-flash` | gemini | 1 000 000 | 2 | 0.50 | 3.00 |
| `gemini-2.5-flash` | gemini | 1 000 000 | 2 | 0.30 | 2.50 |
| `gemini-3.1-flash-lite` | gemini | 1 000 000 | 1 | 0.25 | 1.50 |

> Gemini prices shown are for prompts ≤ 200k tokens.
> `gemini-3.1-pro` and `gemini-2.5-pro` apply higher rates above that threshold.

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
