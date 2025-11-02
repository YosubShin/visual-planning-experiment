"""Evaluation metrics for FrozenLake-style planning tasks."""
from __future__ import annotations

from typing import Iterable, Sequence

from .utils import Action, GridPosition, positions_from_actions


def exact_match(predicted: Sequence[str], optimal: Sequence[str]) -> float:
    """Return ``1.0`` when *predicted* matches *optimal* exactly, ``0.0`` otherwise."""

    return float(list(predicted) == list(optimal))


def _find_start(layout: Sequence[str]) -> GridPosition:
    for r, row in enumerate(layout):
        for c, cell in enumerate(row):
            if cell == "S":
                return GridPosition(r, c)
    raise ValueError("Layout is missing a start position")


def progress_rate(
    predicted_actions: Iterable[str],
    optimal_action_sequences: Sequence[Sequence[str]],
    layout: Sequence[str],
) -> float:
    """Return the ratio of consecutive correct moves along any optimal trajectory."""

    if not optimal_action_sequences:
        return 0.0
    start = _find_start(layout)
    pred_actions: list[Action] = []
    for action in predicted_actions:
        try:
            pred_actions.append(Action(action))
        except ValueError:
            break

    predicted_positions = positions_from_actions(start, pred_actions, layout)[1:]

    best_ratio = 0.0
    for optimal_actions in optimal_action_sequences:
        optimal_steps: list[Action] = []
        for action in optimal_actions:
            try:
                optimal_steps.append(Action(action))
            except ValueError:
                optimal_steps = []
                break

        if not optimal_steps:
            continue

        optimal_positions = positions_from_actions(start, optimal_steps, layout)[1:]
        if not optimal_positions:
            continue

        prefix = 0
        for predicted, optimal in zip(predicted_positions, optimal_positions):
            if predicted != optimal:
                break
            prefix += 1

        best_ratio = max(best_ratio, prefix / len(optimal_positions))

    return best_ratio


def invalid_action_rate(
    predicted_actions: Iterable[str],
    layout: Sequence[str],
) -> float:
    """Compute the fraction of invalid actions in *predicted_actions*."""

    size = len(layout)
    start_coords = _find_start(layout)

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


def empty_prediction_rate(predicted_actions: Sequence[str]) -> float:
    """Return ``1.0`` when no actions are predicted, ``0.0`` otherwise."""

    return 1.0 if len(predicted_actions) == 0 else 0.0


__all__ = ["exact_match", "progress_rate", "invalid_action_rate", "empty_prediction_rate"]
