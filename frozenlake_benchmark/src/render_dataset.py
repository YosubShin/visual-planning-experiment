"""Utilities to generate ASCII and image renderings for existing datasets."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

from .render_ascii import save_ascii
from .render_image import save_image


def load_records(path: Path) -> List[dict]:
    """Load JSON Lines records from *path*."""

    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            records.append(json.loads(text))
    return records


def render_records(
    records: Iterable[dict],
    *,
    output_dir: Path,
    relative_root: Path | None = None,
    write_ascii: bool = True,
    write_images: bool = True,
) -> List[dict]:
    """Render *records* to ``output_dir`` and return updated metadata."""

    relative_root = relative_root or output_dir
    counters: dict[int, int] = defaultdict(int)
    updated: List[dict] = []

    for record in records:
        layout = record["layout"]
        size = record.get("grid_size", len(layout))
        index = counters[size]
        counters[size] += 1

        updated_record = dict(record)
        if write_ascii:
            ascii_path = output_dir / f"ascii_{size}_{index:05d}.txt"
            save_ascii(layout, ascii_path)
            updated_record["ascii_path"] = str(ascii_path.relative_to(relative_root))

        if write_images:
            image_path = output_dir / f"img_{size}_{index:05d}.png"
            save_image(layout, image_path)
            updated_record["image_path"] = str(image_path.relative_to(relative_root))
        updated.append(updated_record)
    return updated


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("frozenlake_benchmark/data/test.jsonl"),
        help="Dataset JSONL file to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frozenlake_benchmark/data/render/test"),
        help="Directory where renderings will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of examples to render.",
    )
    parser.add_argument(
        "--skip-ascii",
        action="store_true",
        help="Skip writing ASCII renderings.",
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip writing PNG renderings.",
    )
    parser.add_argument(
        "--relative-root",
        type=Path,
        default=None,
        help=(
            "Optional root used to compute relative paths stored in the metadata. "
            "Defaults to the output directory."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    records = load_records(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relative_root = args.relative_root
    if relative_root is None:
        dataset_parent = args.dataset.parent
        if dataset_parent.parent != dataset_parent:
            relative_root = dataset_parent.parent
        else:
            relative_root = args.output_dir
    render_records(
        records,
        output_dir=args.output_dir,
        relative_root=relative_root,
        write_ascii=not args.skip_ascii,
        write_images=not args.skip_image,
    )


if __name__ == "__main__":
    main()
