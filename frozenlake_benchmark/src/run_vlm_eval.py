"""Evaluate Qwen models on the FrozenLake benchmark."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .metrics import exact_match, invalid_action_rate, progress_rate
from .render_ascii import ascii_board


try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - optional dependency
    InferenceClient = None  # type: ignore


@dataclass
class EvaluationResult:
    layout: Sequence[str]
    optimal_actions: Sequence[str]
    optimal_action_sequences: Sequence[Sequence[str]]
    predicted_actions: Sequence[str]
    em: float
    pr: float
    iar: float
    raw_response: str


class MockPlanner:
    """Fallback planner that always predicts the optimal actions."""

    def predict(self, record: dict, *_: object) -> str:
        return ", ".join(record["optimal_actions"])


class HuggingFacePlanner:
    """Wrapper around Hugging Face inference endpoints."""

    def __init__(self, model: str, token: str | None, variant: str) -> None:
        if InferenceClient is None:
            raise RuntimeError("huggingface_hub is required for remote inference")
        self.client = InferenceClient(model=model, token=token)
        self.variant = variant

    def predict(self, record: dict, prompt: str) -> str:
        if self.variant == "ascii":
            messages = [
                {"role": "system", "content": "You are an expert grid-world planner."},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat_completion(messages=messages, max_tokens=256)
            return response.choices[0].message["content"]  # type: ignore[index]
        if self.variant == "image":
            raise NotImplementedError(
                "Image-only evaluation requires a local VLM runtime; consider exporting the dataset "
                "and running evaluation in an environment with GPU access."
            )
        raise ValueError(f"Unknown variant: {self.variant}")


PROMPT_TEMPLATE = """You are in a grid world.\nThe symbols are:\nS = start, G = goal, H = hole, F = frozen safe tile.\nYou can move UP, DOWN, LEFT, RIGHT. Avoid holes. Reach the goal.\n\nGrid:\n{grid}\n\nWhat is the sequence of moves from S to G? Respond as a comma-separated list of moves."""


def parse_actions(text: str) -> List[str]:
    """Convert a model completion to a list of canonical actions."""

    cleaned = text.replace("->", "").replace("\n", " ")
    cleaned = cleaned.replace("[", "").replace("]", "")
    cleaned = cleaned.replace("(", "").replace(")", "")
    segments = [segment.strip().upper() for segment in cleaned.split(",")]
    actions = []
    for segment in segments:
        if not segment:
            continue
        token = segment.split()[0]
        if token in {"UP", "DOWN", "LEFT", "RIGHT"}:
            actions.append(token)
    return actions


def load_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def evaluate(
    planner: MockPlanner | HuggingFacePlanner,
    records: Sequence[dict],
    variant: str,
) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for record in records:
        if variant == "ascii":
            prompt = PROMPT_TEMPLATE.format(grid=ascii_board(record["layout"]))
        elif variant == "image":
            prompt = "See attached image."
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        raw_response = planner.predict(record, prompt)
        predicted_actions = parse_actions(raw_response)
        optimal_sequences: Sequence[Sequence[str]] = record.get(
            "optimal_action_sequences", [record["optimal_actions"]]
        )
        em = max(exact_match(predicted_actions, seq) for seq in optimal_sequences)
        pr = progress_rate(predicted_actions, record["path_coords"], record["layout"])
        iar = invalid_action_rate(predicted_actions, record["layout"])
        results.append(
            EvaluationResult(
                layout=record["layout"],
                optimal_actions=record["optimal_actions"],
                optimal_action_sequences=optimal_sequences,
                predicted_actions=predicted_actions,
                em=em,
                pr=pr,
                iar=iar,
                raw_response=raw_response,
            )
        )
    return results


def summarize(results: Sequence[EvaluationResult]) -> dict:
    if not results:
        return {"exact_match": 0.0, "progress_rate": 0.0, "invalid_action_rate": 0.0}

    em = sum(result.em for result in results) / len(results)
    pr = sum(result.pr for result in results) / len(results)
    iar = sum(result.iar for result in results) / len(results)
    return {"exact_match": em, "progress_rate": pr, "invalid_action_rate": iar}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("frozenlake_benchmark/data/test.jsonl"))
    parser.add_argument(
        "--variant",
        choices=["ascii", "image"],
        default="ascii",
        help="Input modality to evaluate.",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "huggingface"],
        default="mock",
        help="Execution backend. Use 'mock' for local validation or 'huggingface' for remote models.",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Instruct")
    parser.add_argument("--token", default=None)
    parser.add_argument("--limit", type=int, default=32, help="Number of examples to evaluate.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    records = load_records(args.dataset)[: args.limit]
    if not records:
        raise SystemExit(f"No records found in {args.dataset}")

    if args.backend == "mock":
        planner = MockPlanner()
    else:
        planner = HuggingFacePlanner(model=args.model, token=args.token, variant=args.variant)
    results = evaluate(planner, records, args.variant)
    summary = summarize(results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
