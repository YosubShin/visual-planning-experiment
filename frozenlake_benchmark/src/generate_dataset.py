"""Dataset generation for the FrozenLake benchmark."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TYPE_CHECKING

from .utils import (
    GridPosition,
    actions_from_path,
    all_shortest_paths,
    random_layout,
    save_jsonl,
    serialize_layout,
)

if TYPE_CHECKING:  # pragma: no cover - imported lazily when renderings are requested
    from .render_ascii import save_ascii
    from .render_image import save_image

DEFAULT_GRID_SIZES = [3, 4, 5, 6]
DEFAULT_TRAIN = 1000
DEFAULT_TEST = 250


def generate_examples(
    *,
    size: int,
    count: int,
    rng: random.Random,
    hole_prob: float,
    render_dir: Path | None,
    relative_root: Path | None,
    save_renderings: bool,
    prohibited: set[str] | None = None,
    max_attempts: int | None = None,
) -> List[Dict]:
    """Generate *count* valid examples for a given grid *size*."""

    examples: List[Dict] = []
    seen_layouts = set(prohibited or set())
    attempts = 0
    max_attempts = max_attempts or count * 1000

    while len(examples) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Unable to sample sufficient unique layouts. "
                "Consider reducing the requested count or allowing duplicates."
            )
        start = GridPosition(rng.randrange(size), rng.randrange(size))
        goal = GridPosition(rng.randrange(size), rng.randrange(size))
        if start == goal:
            continue
        layout = random_layout(size, hole_prob, start, goal, rng)
        serialized = serialize_layout(layout)
        if serialized in seen_layouts:
            continue
        paths = all_shortest_paths(layout, start, goal)
        if not paths:
            continue
        seen_layouts.add(serialized)

        path_action_pairs: List[tuple[List[str], List[List[int]]]] = []
        unique_actions: set[tuple[str, ...]] = set()
        for path in paths:
            actions = tuple(action.value for action in actions_from_path(path))
            if actions in unique_actions:
                continue
            unique_actions.add(actions)
            path_action_pairs.append((list(actions), [pos.to_list() for pos in path]))

        # Deterministically order the action sequences by lexicographic action names.
        path_action_pairs.sort(key=lambda item: item[0])
        canonical_actions, canonical_path = path_action_pairs[0]
        action_sequences = [actions for actions, _ in path_action_pairs]

        idx = len(examples)
        record = {
            "grid_size": size,
            "layout": layout,
            "hole_prob": hole_prob,
            "start": list(start),
            "goal": list(goal),
            "optimal_actions": canonical_actions,
            "optimal_action_sequences": action_sequences,
            "path_coords": canonical_path,
        }

        if save_renderings:
            if render_dir is None or relative_root is None:  # pragma: no cover - sanity guard
                raise ValueError("render_dir and relative_root are required when save_renderings=True")
            from .render_ascii import save_ascii  # local import to avoid optional dependency at runtime
            from .render_image import save_image

            ascii_path = render_dir / f"ascii_{size}_{idx:05d}.txt"
            image_path = render_dir / f"img_{size}_{idx:05d}.png"
            save_ascii(layout, ascii_path)
            save_image(layout, image_path)
            record["ascii_path"] = str(ascii_path.relative_to(relative_root))
            record["image_path"] = str(image_path.relative_to(relative_root))

        examples.append(record)
    return examples


def partition_dataset(
    *,
    sizes: Sequence[int],
    train_count: int,
    test_count: int,
    seed: int,
    output_dir: Path,
    hole_prob: float,
    save_renderings: bool,
    train_counts: Dict[int, int] | None = None,
    test_counts: Dict[int, int] | None = None,
    max_attempts: int | None = None,
) -> None:
    """Generate and persist the train/test splits."""

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = output_dir / "render" if save_renderings else None
    if save_renderings:
        assert render_dir is not None  # for the type-checker
        render_dir.mkdir(parents=True, exist_ok=True)
    relative_root = output_dir.parent if save_renderings else None

    train_records: List[Dict] = []
    test_records: List[Dict] = []

    for size in sizes:
        train_target = (train_counts or {}).get(size, train_count)
        test_target = (test_counts or {}).get(size, test_count)
        train_examples = generate_examples(
            size=size,
            count=train_target,
            rng=rng,
            hole_prob=hole_prob,
            render_dir=render_dir,
            relative_root=relative_root,
            save_renderings=save_renderings,
            max_attempts=max_attempts,
        )
        prohibited = {serialize_layout(record["layout"]) for record in train_examples}
        test_examples = generate_examples(
            size=size,
            count=test_target,
            rng=rng,
            hole_prob=hole_prob,
            render_dir=render_dir,
            relative_root=relative_root,
            save_renderings=save_renderings,
            prohibited=prohibited,
            max_attempts=max_attempts,
        )
        train_records.extend(train_examples)
        test_records.extend(test_examples)

    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(test_records, output_dir / "test.jsonl")


def parse_overrides(pairs: Sequence[str] | None) -> Dict[int, int]:
    overrides: Dict[int, int] = {}
    if not pairs:
        return overrides
    for pair in pairs:
        if ":" not in pair:
            raise argparse.ArgumentTypeError(
                f"Invalid override '{pair}'. Expected format SIZE:COUNT (e.g., 4:250)."
            )
        size_str, count_str = pair.split(":", maxsplit=1)
        try:
            size = int(size_str)
            count = int(count_str)
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(
                f"Invalid override '{pair}'. Expected integers for size and count."
            ) from exc
        if size <= 0 or count < 0:
            raise argparse.ArgumentTypeError(
                f"Invalid override '{pair}'. Size must be positive and count non-negative."
            )
        overrides[size] = count
    return overrides


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("frozenlake_benchmark/data"))
    parser.add_argument("--grid-sizes", type=int, nargs="*", default=DEFAULT_GRID_SIZES)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST)
    parser.add_argument("--hole-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-renderings",
        action="store_true",
        help="Persist ASCII and PNG renderings alongside the JSONL files.",
    )
    parser.add_argument(
        "--train-counts",
        type=str,
        nargs="*",
        default=None,
        help="Override train counts per grid size using SIZE:COUNT pairs (e.g., 3:120 4:300).",
    )
    parser.add_argument(
        "--test-counts",
        type=str,
        nargs="*",
        default=None,
        help="Override test counts per grid size using SIZE:COUNT pairs.",
    )
    args = parser.parse_args(argv)
    try:
        train_counts = parse_overrides(args.train_counts)
        test_counts = parse_overrides(args.test_counts)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    args.train_counts = train_counts or None
    args.test_counts = test_counts or None
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    partition_dataset(
        sizes=args.grid_sizes,
        train_count=args.train_count,
        test_count=args.test_count,
        seed=args.seed,
        output_dir=args.output_dir,
        hole_prob=args.hole_prob,
        save_renderings=args.save_renderings,
        train_counts=args.train_counts,
        test_counts=args.test_counts,
    )


if __name__ == "__main__":
    main()
