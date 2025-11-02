"""Evaluate Qwen models on the FrozenLake benchmark."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .metrics import exact_match, invalid_action_rate, progress_rate
from .render_ascii import ascii_board


try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - optional dependency
    InferenceClient = None  # type: ignore

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForVision2Seq,
        AutoProcessor,
        AutoTokenizer,
    )
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoModelForVision2Seq = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoTokenizer = None  # type: ignore


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


class TransformersPlanner:
    """Local planner backed by `transformers` models."""

    def __init__(
        self,
        model: str,
        variant: str,
        max_new_tokens: int,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "transformers and a working torch installation are required for local inference"
            )
        if AutoModelForCausalLM is None and AutoModelForVision2Seq is None:
            raise RuntimeError("transformers installation is missing model classes required for inference")
        if variant != "ascii":
            raise NotImplementedError("Local image-based evaluation is not supported.")

        self.variant = variant
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.uses_processor = False
        self.processor = None

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = None
        tokenizer_exc: Exception | None = None
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            except Exception as exc:  # pragma: no cover - network/dependency issues
                tokenizer_exc = exc
                self.tokenizer = None

        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        self.model = None
        last_exc: Exception | None = None
        if self.tokenizer is not None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
            except ValueError as exc:
                last_exc = exc
                self.model = None
            except Exception as exc:  # pragma: no cover - network/dependency issues
                raise RuntimeError(
                    f"Failed to load model weights for '{model}'. Ensure the checkpoint is available locally."
                ) from exc
        if self.model is None:
            if AutoModelForVision2Seq is None:
                source_exc = last_exc or tokenizer_exc
                raise RuntimeError(
                    f"Failed to load model weights for '{model}'. Ensure the checkpoint is available locally."
                ) from source_exc
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
            except Exception:  # pragma: no cover - optional dependency
                Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment]
            model_cls = Qwen2_5_VLForConditionalGeneration or AutoModelForVision2Seq
            try:
                self.model = model_cls.from_pretrained(model, **load_kwargs)
            except Exception as exc:  # pragma: no cover - dependency issues
                raise RuntimeError(
                    f"Failed to load model weights for '{model}'. Ensure the checkpoint is available locally."
                ) from exc
            if AutoProcessor is not None:
                try:
                    self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
                except Exception:
                    self.processor = None
                else:
                    self.uses_processor = True
            if self.tokenizer is None:
                if self.processor is not None and hasattr(self.processor, "tokenizer"):
                    self.tokenizer = self.processor.tokenizer  # type: ignore[assignment]
                elif tokenizer_exc is not None:
                    raise RuntimeError(
                        f"Failed to load tokenizer for '{model}'. Download the checkpoint locally and provide its path."
                    ) from tokenizer_exc
                else:
                    raise RuntimeError(
                        f"Failed to load tokenizer for '{model}'. Download the checkpoint locally and provide its path."
                    )

        if self.device.type == "cpu":
            self.model.to(self.device)
        self.model.eval()

    def predict(self, record: dict, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert grid-world planner."},
            {"role": "user", "content": prompt},
        ]
        template = None
        use_processor = self.uses_processor and self.processor is not None
        if use_processor:
            try:
                template = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                template = None

        if template is None and self.tokenizer is not None:
            try:
                template = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (AttributeError, ValueError):
                template = None

        if template is None:
            # Fallback for models without chat template support.
            system_prompt = messages[0]["content"]
            user_prompt = messages[1]["content"]
            template = f"{system_prompt}\n\n{user_prompt}"

        if use_processor:
            assert self.processor is not None  # for type-checkers
            inputs = self.processor(
                text=[template], padding=True, return_tensors="pt"
            ).to(self.device)
            tokenizer_like = self.tokenizer
            if tokenizer_like is None and hasattr(self.processor, "tokenizer"):
                tokenizer_like = self.processor.tokenizer  # type: ignore[assignment]
            pad_token_id = (
                getattr(tokenizer_like, "eos_token_id", None)
                if tokenizer_like is not None
                else None
            )
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            }
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            with torch.no_grad():
                generated = self.model.generate(**inputs, **generate_kwargs)
            output_ids = generated[:, inputs["input_ids"].shape[1] :]
            outputs = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return outputs[0].strip() if outputs else ""

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer unavailable for causal language model inference")
        inputs = self.tokenizer(template, return_tensors="pt").to(self.device)
        pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        output_ids = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


PROMPT_TEMPLATE = """Task : Frozen Lake Shortest Path Planning\nYou are given an image of a grid - based environment . In this\nenvironment :\n- An elf marks the starting position .\n- A gift represents the goal .\n- Some cells contain ice holes that are impassable for the elf .\n- The elf can move in one of four directions only : " up " , " down " , " left\n" , or " right ". Each move transitions the elf by one cell in the\ncorresponding absolute direction . Diagonal movement is not\npermitted .\nYour task is to analyze the image and generate the shortest valid\nsequence of actions that moves the elf from the starting position\nto the goal without stepping into any ice holes .\nProvide your final answer enclosed between < ANSWER > and </ ANSWER > , for\nexample : < ANSWER > right up up </ ANSWER >.\n\nGrid:\n{grid}"""


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
        if token in {"up", "down", "left", "right"}:
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
        choices=["mock", "huggingface", "transformers"],
        default="mock",
        help=(
            "Execution backend. Use 'mock' for local validation, 'huggingface' for remote endpoints, "
            "or 'transformers' for offline checkpoints."
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Instruct")
    parser.add_argument("--token", default=None)
    parser.add_argument("--limit", type=int, default=32, help="Number of examples to evaluate.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Number of tokens to generate per sample when using the transformers backend.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL file to persist the evaluation summary.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    records = load_records(args.dataset)[: args.limit]
    if not records:
        raise SystemExit(f"No records found in {args.dataset}")

    if args.backend == "mock":
        planner = MockPlanner()
    elif args.backend == "transformers":
        planner = TransformersPlanner(
            model=args.model,
            variant=args.variant,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        planner = HuggingFacePlanner(model=args.model, token=args.token, variant=args.variant)
    results = evaluate(planner, records, args.variant)
    summary = summarize(results)
    print(json.dumps(summary, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": str(args.dataset),
            "backend": args.backend,
            "model": args.model,
            "variant": args.variant,
            "limit": args.limit,
            "summary": summary,
            "records": [asdict(result) for result in results],
        }
        with args.output.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
