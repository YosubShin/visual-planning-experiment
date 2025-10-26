"""Utility functions and helpers for the FrozenLake benchmark."""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


class Action(str, Enum):
    """Enumeration of allowed moves in the grid world."""

    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    @property
    def delta(self) -> Tuple[int, int]:
        """Return the (row, col) delta associated with the action."""

        return {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }[self]


CELL_TYPES = {"S", "G", "H", "F"}


@dataclass(frozen=True)
class GridPosition:
    row: int
    col: int

    def __iter__(self):
        yield self.row
        yield self.col

    def __add__(self, other: Tuple[int, int]) -> "GridPosition":
        dr, dc = other
        return GridPosition(self.row + dr, self.col + dc)

    def to_list(self) -> List[int]:
        return [self.row, self.col]


def serialize_layout(layout: Sequence[str]) -> str:
    """Convert a list of strings representing the layout into a canonical form."""

    return "\n".join(layout)


def deserialize_layout(serialized: str) -> List[str]:
    """Inverse operation of :func:`serialize_layout`."""

    return serialized.split("\n")


def is_inside(position: GridPosition, size: int) -> bool:
    """Return ``True`` when *position* resides inside the board."""

    return 0 <= position.row < size and 0 <= position.col < size


def apply_action(position: GridPosition, action: Action, size: int) -> GridPosition:
    """Apply an action to a position, respecting board boundaries."""

    candidate = position + action.delta
    if not is_inside(candidate, size):
        return position
    return candidate


def positions_from_actions(
    start: GridPosition, actions: Iterable[Action], layout: Sequence[str]
) -> List[GridPosition]:
    """Generate the visited positions when executing *actions* starting from *start*."""

    size = len(layout)
    positions = [start]
    current = start
    for action in actions:
        next_pos = apply_action(current, action, size)
        cell = layout[next_pos.row][next_pos.col]
        positions.append(next_pos)
        if cell == "H":
            break
        current = next_pos
        if cell == "G":
            break
    return positions


def bfs_shortest_path(layout: Sequence[str], start: GridPosition, goal: GridPosition) -> Optional[List[GridPosition]]:
    """Return the shortest path using BFS or ``None`` if no path exists."""

    size = len(layout)
    queue: deque[GridPosition] = deque([start])
    parents: dict[GridPosition, Optional[GridPosition]] = {start: None}

    while queue:
        node = queue.popleft()
        if node == goal:
            break
        for action in Action:
            neighbor = apply_action(node, action, size)
            if neighbor in parents:
                continue
            if layout[neighbor.row][neighbor.col] == "H":
                continue
            parents[neighbor] = node
            queue.append(neighbor)
    else:
        return None

    path: List[GridPosition] = []
    cursor: Optional[GridPosition] = goal
    while cursor is not None:
        path.append(cursor)
        cursor = parents[cursor]
    path.reverse()
    return path


def all_shortest_paths(
    layout: Sequence[str],
    start: GridPosition,
    goal: GridPosition,
) -> List[List[GridPosition]]:
    """Return all shortest paths from *start* to *goal* (may be empty)."""

    size = len(layout)
    queue: deque[GridPosition] = deque([start])
    distances: dict[GridPosition, int] = {start: 0}
    parents: dict[GridPosition, List[GridPosition]] = {start: []}

    while queue:
        node = queue.popleft()
        if node == goal:
            continue
        for action in Action:
            neighbor = apply_action(node, action, size)
            if layout[neighbor.row][neighbor.col] == "H":
                continue
            new_distance = distances[node] + 1
            if neighbor not in distances:
                distances[neighbor] = new_distance
                parents[neighbor] = [node]
                queue.append(neighbor)
            elif new_distance == distances[neighbor]:
                parents.setdefault(neighbor, [])
                parents[neighbor].append(node)

    if goal not in distances:
        return []

    for node, parent_list in parents.items():
        parents[node] = sorted(parent_list, key=lambda pos: (pos.row, pos.col))

    paths: List[List[GridPosition]] = []

    def backtrack(current: GridPosition, suffix: List[GridPosition]) -> None:
        if current == start:
            paths.append([start, *suffix])
            return
        for parent in parents.get(current, []):
            backtrack(parent, [current, *suffix])

    backtrack(goal, [])
    return paths


def actions_from_path(path: Sequence[GridPosition]) -> List[Action]:
    """Transform a path into an action sequence."""

    actions: List[Action] = []
    for prev, nxt in zip(path, path[1:]):
        dr = nxt.row - prev.row
        dc = nxt.col - prev.col
        for action in Action:
            if action.delta == (dr, dc):
                actions.append(action)
                break
        else:
            raise ValueError(f"Invalid step from {prev} to {nxt}")
    return actions


def random_layout(
    size: int,
    hole_prob: float,
    start: GridPosition,
    goal: GridPosition,
    rng: random.Random,
) -> List[str]:
    """Sample a random map subject to validity constraints."""

    layout = []
    for r in range(size):
        row_chars = []
        for c in range(size):
            if GridPosition(r, c) == start:
                row_chars.append("S")
            elif GridPosition(r, c) == goal:
                row_chars.append("G")
            else:
                row_chars.append("H" if rng.random() < hole_prob else "F")
        layout.append("".join(row_chars))
    return layout


def ensure_directory(path: Path) -> None:
    """Create the directory of *path* if necessary."""

    path.mkdir(parents=True, exist_ok=True)


def save_jsonl(records: Sequence[dict], path: Path) -> None:
    """Save *records* to *path* in JSON Lines format."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


__all__ = [
    "Action",
    "GridPosition",
    "all_shortest_paths",
    "actions_from_path",
    "apply_action",
    "bfs_shortest_path",
    "ensure_directory",
    "positions_from_actions",
    "random_layout",
    "save_jsonl",
    "serialize_layout",
    "deserialize_layout",
]
