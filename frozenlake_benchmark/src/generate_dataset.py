"""Dataset generation for the FrozenLake benchmark."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .render_ascii import save_ascii
from .render_image import save_image
from .utils import (
    GridPosition,
    actions_from_path,
    bfs_shortest_path,
    random_layout,
    save_jsonl,
    serialize_layout,
)

DEFAULT_GRID_SIZES = [3, 4, 5, 6]
DEFAULT_TRAIN = 1000
DEFAULT_TEST = 250


def generate_examples(
    *,
    size: int,
    count: int,
    rng: random.Random,
    hole_prob: float,
    render_dir: Path,
    relative_root: Path,
    prohibited: set[str] | None = None,
) -> List[Dict]:
    """Generate *count* valid examples for a given grid *size*."""

    start = GridPosition(0, 0)
    goal = GridPosition(size - 1, size - 1)

    examples: List[Dict] = []
    seen_layouts = set(prohibited or set())

    while len(examples) < count:
        layout = random_layout(size, hole_prob, start, goal, rng)
        serialized = serialize_layout(layout)
        if serialized in seen_layouts:
            continue
        path = bfs_shortest_path(layout, start, goal)
        if not path:
            continue
        seen_layouts.add(serialized)
        actions = [action.value for action in actions_from_path(path)]

        idx = len(examples)
        ascii_path = render_dir / f"ascii_{size}_{idx:05d}.txt"
        image_path = render_dir / f"img_{size}_{idx:05d}.png"
        save_ascii(layout, ascii_path)
        save_image(layout, image_path)

        examples.append(
            {
                "grid_size": size,
                "layout": layout,
                "hole_prob": hole_prob,
                "start": list(start),
                "goal": list(goal),
                "optimal_actions": actions,
                "path_coords": [pos.to_list() for pos in path],
                "ascii_path": str(ascii_path.relative_to(relative_root)),
                "image_path": str(image_path.relative_to(relative_root)),
            }
        )
    return examples


def partition_dataset(
    *,
    sizes: Sequence[int],
    train_count: int,
    test_count: int,
    seed: int,
    output_dir: Path,
    hole_prob: float,
) -> None:
    """Generate and persist the train/test splits."""

    rng = random.Random(seed)
    render_dir = output_dir / "render"
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    relative_root = output_dir.parent

    train_records: List[Dict] = []
    test_records: List[Dict] = []

    for size in sizes:
        train_examples = generate_examples(
            size=size,
            count=train_count,
            rng=rng,
            hole_prob=hole_prob,
            render_dir=render_dir,
            relative_root=relative_root,
        )
        prohibited = {serialize_layout(record["layout"]) for record in train_examples}
        test_examples = generate_examples(
            size=size,
            count=test_count,
            rng=rng,
            hole_prob=hole_prob,
            render_dir=render_dir,
            relative_root=relative_root,
            prohibited=prohibited,
        )
        train_records.extend(train_examples)
        test_records.extend(test_examples)

    save_jsonl(train_records, output_dir / "train.jsonl")
    save_jsonl(test_records, output_dir / "test.jsonl")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("frozenlake_benchmark/data"))
    parser.add_argument("--grid-sizes", type=int, nargs="*", default=DEFAULT_GRID_SIZES)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST)
    parser.add_argument("--hole-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    partition_dataset(
        sizes=args.grid_sizes,
        train_count=args.train_count,
        test_count=args.test_count,
        seed=args.seed,
        output_dir=args.output_dir,
        hole_prob=args.hole_prob,
    )


if __name__ == "__main__":
    main()
