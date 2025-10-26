"""Image rendering utilities for FrozenLake layouts."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping, Sequence, Tuple

from PIL import Image, ImageDraw

DEFAULT_COLORS: Mapping[str, Tuple[int, int, int]] = {
    "S": (55, 126, 184),
    "G": (77, 175, 74),
    "H": (228, 26, 28),
    "F": (255, 255, 204),
}


def render_image(
    layout: Sequence[str],
    cell_size: int = 64,
    color_map: MutableMapping[str, Tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Render *layout* as a PIL image."""

    colors = dict(DEFAULT_COLORS)
    if color_map:
        colors.update(color_map)

    size = len(layout)
    image = Image.new("RGB", (cell_size * size, cell_size * size), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for r, row in enumerate(layout):
        for c, cell in enumerate(row):
            top_left = (c * cell_size, r * cell_size)
            bottom_right = ((c + 1) * cell_size, (r + 1) * cell_size)
            fill = colors.get(cell, (0, 0, 0))
            draw.rectangle([top_left, bottom_right], fill=fill, outline=(0, 0, 0))
    return image


def save_image(
    layout: Sequence[str],
    path: Path,
    cell_size: int = 64,
    color_map: MutableMapping[str, Tuple[int, int, int]] | None = None,
) -> None:
    """Render and write the image for *layout* to *path*."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image = render_image(layout, cell_size=cell_size, color_map=color_map)
    image.save(path)


__all__ = ["render_image", "save_image", "DEFAULT_COLORS"]
