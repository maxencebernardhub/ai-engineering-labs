"""Compare several OpenAI models on the same prompt.

This lab compares models on four practical dimensions:
1. Quality of the answer content
2. Latency
3. Estimated cost
4. JSON stability

Important design choice:
- This lab intentionally asks the models to return JSON via prompting alone.
- It does not use strict structured outputs for the benchmark itself.
- That makes the JSON stability test meaningful, because the model must follow
  the requested JSON format without schema enforcement from the API.

The comparison is heuristic, not scientific. It is meant to help you explore
trade-offs between stronger and cheaper models in a hands-on way.

Official pricing source used when writing this lab:
https://developers.openai.com/api/docs/pricing
"""

from __future__ import annotations

import ast
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses.response_usage import ResponseUsage
from pydantic import BaseModel, ValidationError

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

RUNS_PER_MODEL = 3

MODEL_ORDER = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]

PRICING_PER_1M_TOKENS = {
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
}

COMPARISON_PROMPT = """
Answer the following question as valid JSON only.

Question:
Explain the difference between a Python list and a Python tuple, when to use
each one, and provide one very small example.

Return exactly this JSON shape:
{
  "summary": "string",
  "differences": [
    {"topic": "string", "list": "string", "tuple": "string"},
    {"topic": "string", "list": "string", "tuple": "string"},
    {"topic": "string", "list": "string", "tuple": "string"}
  ],
  "example_code": "string",
  "best_choice_rule": "string",
  "common_mistake": "string"
}

Rules:
- Return raw JSON only, with no markdown fences.
- Use exactly these keys in exactly this order.
- `differences` must contain exactly 3 items.
- Each `topic` must be one of: `mutability`, `performance`, `use_case`.
- `summary` must be 18 words or fewer.
- `best_choice_rule` must be 16 words or fewer.
- `common_mistake` must mention either `hashable` or `dictionary key`.
- `example_code` must be valid Python and contain exactly two assignments:
  one list and one tuple.
- Mention mutability clearly and be technically precise.
""".strip()


class DifferenceItem(BaseModel):
    """Represent one structured comparison point."""

    topic: str
    list: str
    tuple: str


class ComparisonPayload(BaseModel):
    """Represent the stricter JSON shape expected from every model."""

    summary: str
    differences: list[DifferenceItem]
    example_code: str
    best_choice_rule: str
    common_mistake: str


@dataclass
class SingleRunResult:
    """Store benchmark data for one model call."""

    model: str
    latency_seconds: float
    estimated_cost_usd: float
    json_valid: bool
    quality_score: float
    raw_output: str
    usage: ResponseUsage
    parsed_payload: ComparisonPayload | None


def require_api_key() -> None:
    """Fail early with a clear error if the API key is missing."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def print_section(title: str) -> None:
    """Print a readable separator for each scenario."""

    print(f"\n{'=' * 18} {title} {'=' * 18}")


def estimate_cost_usd(model: str, usage: ResponseUsage) -> float:
    """Estimate request cost from usage and the model pricing table."""

    pricing = PRICING_PER_1M_TOKENS[model]
    cached_input_tokens = usage.input_tokens_details.cached_tokens
    non_cached_input_tokens = usage.input_tokens - cached_input_tokens

    input_cost = (non_cached_input_tokens / 1_000_000) * pricing["input"]
    cached_input_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]

    return input_cost + cached_input_cost + output_cost


def parse_payload(raw_output: str) -> ComparisonPayload | None:
    """Parse one model output as strict JSON and validate its structure."""

    try:
        payload = json.loads(raw_output)
        return ComparisonPayload.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return None


def word_count(text: str) -> int:
    """Count words with a simple whitespace-based heuristic."""

    return len(text.split())


def has_exact_key_order(raw_output: str) -> bool:
    """Check that the JSON top-level keys match the requested order exactly."""

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return False

    if not isinstance(payload, dict):
        return False

    return list(payload.keys()) == [
        "summary",
        "differences",
        "example_code",
        "best_choice_rule",
        "common_mistake",
    ]


def is_python_example_valid(example_code: str) -> bool:
    """Check that the returned code is valid Python and matches the format rule."""

    try:
        parsed = ast.parse(example_code)
    except SyntaxError:
        return False

    if len(parsed.body) != 2:
        return False

    if not all(isinstance(node, ast.Assign) for node in parsed.body):
        return False

    assignment_values = [
        node.value for node in parsed.body if isinstance(node, ast.Assign)
    ]
    if len(assignment_values) != 2:
        return False

    has_list_assignment = any(
        isinstance(value, ast.List) for value in assignment_values
    )
    has_tuple_assignment = any(
        isinstance(value, ast.Tuple) for value in assignment_values
    )
    return has_list_assignment and has_tuple_assignment


def is_json_compliant(raw_output: str, payload: ComparisonPayload | None) -> bool:
    """Apply stricter JSON compliance checks than basic parsing alone."""

    if payload is None:
        return False

    if not has_exact_key_order(raw_output):
        return False

    if len(payload.differences) != 3:
        return False

    topics = [item.topic for item in payload.differences]
    if topics != ["mutability", "performance", "use_case"]:
        return False

    if word_count(payload.summary) > 18:
        return False

    if word_count(payload.best_choice_rule) > 16:
        return False

    if not contains_any(payload.common_mistake, ["hashable", "dictionary key"]):
        return False

    if not is_python_example_valid(payload.example_code):
        return False

    return True


def contains_any(text: str, terms: list[str]) -> bool:
    """Return whether at least one target term appears in the text."""

    lowered = text.lower()
    return any(term in lowered for term in terms)


def score_quality(payload: ComparisonPayload | None) -> float:
    """Assign a stricter heuristic quality score from 0 to 10."""

    if payload is None:
        return 0.0

    score = 0.0

    if len(payload.differences) == 3:
        score += 1.0

    combined_text = " ".join(
        [
            payload.summary,
            " ".join(
                f"{item.topic} {item.list} {item.tuple}" for item in payload.differences
            ),
            payload.example_code,
            payload.best_choice_rule,
            payload.common_mistake,
        ]
    ).lower()

    if contains_any(combined_text, ["mutable", "immut", "cannot be changed"]):
        score += 1.0

    if contains_any(combined_text, ["list", "[", "append", "modify"]):
        score += 1.0

    if contains_any(combined_text, ["tuple", "(", "fixed", "hashable"]):
        score += 1.0

    if contains_any(combined_text, ["use", "when", "choose", "best"]):
        score += 1.0

    topics = [item.topic for item in payload.differences]
    if topics == ["mutability", "performance", "use_case"]:
        score += 1.0

    if word_count(payload.summary) <= 18:
        score += 1.0

    if word_count(payload.best_choice_rule) <= 16:
        score += 1.0

    if contains_any(payload.common_mistake.lower(), ["hashable", "dictionary key"]):
        score += 1.0

    if is_python_example_valid(payload.example_code):
        score += 2.0

    return score


def run_single_model_call(client: OpenAI, model: str) -> SingleRunResult:
    """Run one benchmark call for a single model."""

    started_at = time.perf_counter()
    response = client.responses.create(
        model=model,
        reasoning={"effort": "minimal"},
        input=COMPARISON_PROMPT,
    )
    latency_seconds = time.perf_counter() - started_at

    parsed_payload = parse_payload(response.output_text)
    quality_score = score_quality(parsed_payload)
    json_valid = is_json_compliant(response.output_text, parsed_payload)

    return SingleRunResult(
        model=model,
        latency_seconds=latency_seconds,
        estimated_cost_usd=estimate_cost_usd(model, response.usage),
        json_valid=json_valid,
        quality_score=quality_score,
        raw_output=response.output_text,
        usage=response.usage,
        parsed_payload=parsed_payload,
    )


def run_showcase_scenario(client: OpenAI) -> None:
    """Run one call per model and show the raw outputs."""

    print_section("Scenario 1 - Single-pass showcase")

    for model in MODEL_ORDER:
        result = run_single_model_call(client, model)

        print(f"\nModel: {model}")
        print(f"- latency_seconds: {result.latency_seconds:.2f}")
        print(f"- estimated_cost_usd: ${result.estimated_cost_usd:.6f}")
        print(f"- json_valid: {result.json_valid}")
        print(f"- quality_score: {result.quality_score:.1f}/10.0")
        print("- raw_output:")
        print(result.raw_output)


def run_benchmark_scenario(client: OpenAI) -> None:
    """Run repeated calls and aggregate the comparison metrics."""

    print_section("Scenario 2 - Repeated benchmark summary")

    print(f"Benchmark configuration: {RUNS_PER_MODEL} runs per model")
    print()
    print(
        "model           avg_quality  json_success  avg_latency_s  "
        "avg_cost_usd  avg_input_tokens  avg_output_tokens"
    )
    print("-" * 96)

    for model in MODEL_ORDER:
        results = [run_single_model_call(client, model) for _ in range(RUNS_PER_MODEL)]

        avg_quality = sum(item.quality_score for item in results) / RUNS_PER_MODEL
        json_success_rate = (
            sum(1 for item in results if item.json_valid) / RUNS_PER_MODEL
        )
        avg_latency = sum(item.latency_seconds for item in results) / RUNS_PER_MODEL
        avg_cost = sum(item.estimated_cost_usd for item in results) / RUNS_PER_MODEL
        avg_input_tokens = (
            sum(item.usage.input_tokens for item in results) / RUNS_PER_MODEL
        )
        avg_output_tokens = (
            sum(item.usage.output_tokens for item in results) / RUNS_PER_MODEL
        )

        print(
            f"{model:<15} "
            f"{avg_quality:>10.2f}  "
            f"{json_success_rate:>11.0%}  "
            f"{avg_latency:>13.2f}  "
            f"{avg_cost:>12.6f}  "
            f"{avg_input_tokens:>16.1f}  "
            f"{avg_output_tokens:>17.1f}"
        )


def run_interpretation_scenario() -> None:
    """Print a reading guide for the benchmark output."""

    print_section("Scenario 3 - How to read the results")
    print("- quality: heuristic score out of 10 with stricter content checks")
    print("- json_success: strict compliance, not just parsable JSON")
    print("- avg_latency_s: observed wall-clock latency from the client side")
    print("- avg_cost_usd: estimated from token usage and the published pricing")
    print("- avg_input_tokens / avg_output_tokens: useful for cost reasoning")


def main() -> None:
    """Run all model-comparison scenarios."""

    require_api_key()
    client = OpenAI()

    run_showcase_scenario(client)
    run_benchmark_scenario(client)
    run_interpretation_scenario()


if __name__ == "__main__":
    main()
