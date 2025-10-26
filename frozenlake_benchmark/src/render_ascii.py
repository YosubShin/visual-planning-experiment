"""ASCII rendering utilities for FrozenLake layouts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .utils import Action, GridPosition, apply_action


def ascii_board(layout: Sequence[str]) -> str:
    """Return a multi-line ASCII representation of *layout*."""

    return "\n".join(" ".join(row) for row in layout)


def render_with_agent(
    layout: Sequence[str],
    actions: Iterable[str],
) -> str:
    """Render the layout while animating the agent following *actions*."""

    size = len(layout)
    start = None
    for r, row in enumerate(layout):
        for c, cell in enumerate(row):
            if cell == "S":
                start = GridPosition(r, c)
                break
        if start is not None:
            break
    if start is None:
        raise ValueError("Layout is missing a start position")

    grid = [list(row) for row in layout]
    current = start
    for action_str in actions:
        try:
            action = Action(action_str)
        except ValueError:
            continue
        next_pos = apply_action(current, action, size)
        if layout[next_pos.row][next_pos.col] == "H":
            grid[next_pos.row][next_pos.col] = "X"
            break
        current = next_pos
        grid[current.row][current.col] = "A"
    grid[start.row][start.col] = "S"
    return "\n".join(" ".join(row) for row in grid)


def save_ascii(layout: Sequence[str], path: Path) -> None:
    """Write the ASCII board to *path*."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ascii_board(layout) + "\n", encoding="utf-8")


__all__ = ["ascii_board", "render_with_agent", "save_ascii"]
