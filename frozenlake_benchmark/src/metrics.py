"""Evaluation metrics for FrozenLake-style planning tasks."""
from __future__ import annotations

from typing import Iterable, Sequence

from .utils import Action, GridPosition, positions_from_actions


def exact_match(predicted: Sequence[str], optimal: Sequence[str]) -> float:
    """Return ``1.0`` when *predicted* matches *optimal* exactly, ``0.0`` otherwise."""

    return float(list(predicted) == list(optimal))


def progress_rate(
    predicted_actions: Iterable[str],
    optimal_path: Sequence[Sequence[int]],
    layout: Sequence[str],
) -> float:
    """Compute the progress rate defined as the overlap with the optimal path."""

    if not optimal_path:
        return 0.0
    start = GridPosition(*optimal_path[0])
    optimal_positions = [GridPosition(*coords) for coords in optimal_path]

    pred_actions: list[Action] = []
    for action in predicted_actions:
        try:
            pred_actions.append(Action(action))
        except ValueError:
            break

    visited = positions_from_actions(start, pred_actions, layout)

    overlap = 0
    for expected, actual in zip(optimal_positions, visited):
        if expected == actual:
            overlap += 1
        else:
            break
    return overlap / len(optimal_positions)


def invalid_action_rate(
    predicted_actions: Iterable[str],
    layout: Sequence[str],
) -> float:
    """Compute the fraction of invalid actions in *predicted_actions*."""

    size = len(layout)
    start_coords = None
    for r, row in enumerate(layout):
        for c, cell in enumerate(row):
            if cell == "S":
                start_coords = GridPosition(r, c)
                break
        if start_coords is not None:
            break
    if start_coords is None:
        raise ValueError("Layout is missing a start position")

    current = start_coords
    invalid = 0
    total = 0
    for action_str in predicted_actions:
        total += 1
        try:
            action = Action(action_str)
        except ValueError:
            invalid += 1
            continue
        candidate = current + action.delta
        if not (0 <= candidate.row < size and 0 <= candidate.col < size):
            invalid += 1
            continue
        cell = layout[candidate.row][candidate.col]
        if cell == "H":
            invalid += 1
            current = candidate
            break
        current = candidate
    return invalid / total if total else 0.0


__all__ = ["exact_match", "progress_rate", "invalid_action_rate"]
