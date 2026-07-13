from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

DamagePredictionFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class RecoveryStep:
    voxels: np.ndarray
    predicted_damage: np.ndarray
    confidence: np.ndarray
    added_mask: np.ndarray
    missing_count: Optional[int]
    extra_count: Optional[int] = None
    total_added_count: int = 0


@dataclass
class RecoveryTrajectory:
    steps: List[RecoveryStep]


def recover_damage(
    damaged_voxels: np.ndarray,
    predict_fn: DamagePredictionFn,
    original_mask: Optional[np.ndarray] = None,
    recovery_steps: int = 24,
    confidence_threshold: float = 0.0,
    confidence_window: int = 12,
    confidence_required: int = 6,
    max_additions_per_step: Optional[int] = None,
    constrain_to_original: bool = True,
    show_progress: bool = True,
    no_progress_patience: int = 0,
    extra_steps_after_complete: int = 0,
    consensus_min_votes: int = 2,
    single_vote_confidence_threshold: float = 0.99,
) -> RecoveryTrajectory:
    """Iteratively predict damage directions and add repaired voxels."""
    if confidence_window <= 0:
        raise ValueError("confidence_window must be positive.")
    if confidence_required <= 0:
        raise ValueError("confidence_required must be positive.")
    if confidence_required > confidence_window:
        raise ValueError("confidence_required cannot exceed confidence_window.")
    if no_progress_patience < 0:
        raise ValueError("no_progress_patience must be non-negative.")
    if extra_steps_after_complete < 0:
        raise ValueError("extra_steps_after_complete must be non-negative.")
    if consensus_min_votes <= 0:
        raise ValueError("consensus_min_votes must be positive.")
    if not 0.0 <= single_vote_confidence_threshold <= 1.0:
        raise ValueError("single_vote_confidence_threshold must be in [0, 1].")

    current = damaged_voxels.astype(np.uint8).copy()
    initial = current.copy()
    original = None
    if original_mask is not None:
        original = original_mask.astype(np.uint8)

    steps = []
    added_mask = np.zeros_like(current, dtype=np.uint8)
    vote_history = deque(maxlen=confidence_window)
    no_progress_count = 0
    complete_steps = 0
    iterator = recovery_iterator(recovery_steps, show_progress)
    try:
        for _ in iterator:
            predicted_damage, confidence = predict_fn(current)
            missing_count, extra_count, total_added_count = recovery_counts(
                original,
                initial,
                current,
            )
            update_recovery_progress(
                iterator,
                show_progress,
                current,
                missing_count,
                extra_count,
                total_added_count,
            )

            steps.append(
                RecoveryStep(
                    voxels=current.copy(),
                    predicted_damage=predicted_damage.copy(),
                    confidence=confidence.copy(),
                    added_mask=added_mask.copy(),
                    missing_count=missing_count,
                    extra_count=extra_count,
                    total_added_count=total_added_count,
                )
            )

            if missing_count == 0:
                if complete_steps >= extra_steps_after_complete:
                    break
                complete_steps += 1
            else:
                complete_steps = 0

            raw_candidates = candidate_repairs(
                current=current,
                predicted_damage=predicted_damage,
                confidence=confidence,
                original_mask=original,
                confidence_threshold=confidence_threshold,
                constrain_to_original=constrain_to_original,
            )
            candidates = consensus_repair_candidates(
                raw_candidates,
                consensus_min_votes=consensus_min_votes,
                single_vote_confidence_threshold=single_vote_confidence_threshold,
            )
            vote_mask = np.zeros_like(current, dtype=bool)
            for _, _, target in candidates:
                vote_mask[target] = True
            vote_history.append(vote_mask)

            if confidence_required > 1:
                vote_counts = np.zeros_like(current, dtype=np.uint16)
                for previous_votes in vote_history:
                    vote_counts += previous_votes
                candidates = [
                    (float(vote_counts[target]), confidence_value, target)
                    for _, confidence_value, target in candidates
                    if vote_counts[target] >= confidence_required
                ]

            current, added_mask, added_count = apply_repair_candidates(
                current=current,
                candidates=candidates,
                max_additions=max_additions_per_step,
            )
            (
                updated_missing_count,
                updated_extra_count,
                updated_total_added_count,
            ) = recovery_counts(original, initial, current)
            update_recovery_progress(
                iterator,
                show_progress,
                current,
                updated_missing_count,
                updated_extra_count,
                updated_total_added_count,
            )

            if updated_missing_count == 0:
                update_recovery_progress(
                    iterator,
                    show_progress,
                    current,
                    updated_missing_count,
                    updated_extra_count,
                    updated_total_added_count,
                )
                steps.append(
                    RecoveryStep(
                        voxels=current.copy(),
                        predicted_damage=predicted_damage.copy(),
                        confidence=confidence.copy(),
                        added_mask=added_mask.copy(),
                        missing_count=updated_missing_count,
                        extra_count=updated_extra_count,
                        total_added_count=updated_total_added_count,
                    )
                )
                if extra_steps_after_complete == 0:
                    break
                complete_steps = 1
            if added_count == 0:
                no_progress_count += 1
                if no_progress_count <= no_progress_patience:
                    continue
                break
            no_progress_count = 0
    finally:
        if show_progress and hasattr(iterator, "close"):
            iterator.close()

    return RecoveryTrajectory(steps=steps)


def recovery_counts(
    original: Optional[np.ndarray],
    initial: np.ndarray,
    current: np.ndarray,
) -> Tuple[Optional[int], Optional[int], int]:
    missing_count = count_missing(original, current)
    extra_count = count_extra(original, current)
    total_added_count = int(((current == 1) & (initial == 0)).sum())
    return missing_count, extra_count, total_added_count


def count_missing(original: Optional[np.ndarray], current: np.ndarray) -> Optional[int]:
    if original is None:
        return None
    return int(((original == 1) & (current == 0)).sum())


def count_extra(original: Optional[np.ndarray], current: np.ndarray) -> Optional[int]:
    if original is None:
        return None
    return int(((original == 0) & (current == 1)).sum())


def update_recovery_progress(
    iterator,
    show_progress: bool,
    current: np.ndarray,
    missing_count: Optional[int],
    extra_count: Optional[int],
    total_added_count: int,
):
    if not show_progress or not hasattr(iterator, "set_postfix"):
        return

    iterator.set_postfix(
        voxels=int(current.sum()),
        missing=missing_count if missing_count is not None else "unknown",
        extra=extra_count if extra_count is not None else "unknown",
        added=total_added_count,
    )


def recovery_iterator(recovery_steps: int, show_progress: bool):
    iterator = range(recovery_steps + 1)
    if not show_progress:
        return iterator

    from tqdm import tqdm

    return tqdm(iterator, desc="Recovering", unit="step")


def recover_from_prediction(
    current: np.ndarray,
    predicted_damage: np.ndarray,
    confidence: np.ndarray,
    original_mask: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.0,
    max_additions: Optional[int] = None,
    constrain_to_original: bool = True,
    consensus_min_votes: int = 2,
    single_vote_confidence_threshold: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, int]:
    candidates = consensus_repair_candidates(
        candidate_repairs(
            current=current,
            predicted_damage=predicted_damage,
            confidence=confidence,
            original_mask=original_mask,
            confidence_threshold=confidence_threshold,
            constrain_to_original=constrain_to_original,
        ),
        consensus_min_votes=consensus_min_votes,
        single_vote_confidence_threshold=single_vote_confidence_threshold,
    )
    return apply_repair_candidates(current, candidates, max_additions)


def consensus_repair_candidates(
    candidates: List[Tuple[float, Tuple[int, int, int]]],
    consensus_min_votes: int = 2,
    single_vote_confidence_threshold: float = 0.99,
) -> List[Tuple[float, float, Tuple[int, int, int]]]:
    target_votes = {}
    for confidence_value, target in candidates:
        vote_count, max_confidence = target_votes.get(target, (0, 0.0))
        target_votes[target] = (
            vote_count + 1,
            max(max_confidence, float(confidence_value)),
        )

    return [
        (float(vote_count), max_confidence, target)
        for target, (vote_count, max_confidence) in target_votes.items()
        if vote_count >= consensus_min_votes
        or max_confidence >= single_vote_confidence_threshold
    ]


def candidate_repairs(
    current: np.ndarray,
    predicted_damage: np.ndarray,
    confidence: np.ndarray,
    original_mask: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.0,
    constrain_to_original: bool = True,
) -> List[Tuple[float, Tuple[int, int, int]]]:
    candidates = []
    for x, y, z in np.argwhere((current == 1) & (predicted_damage > 0)):
        if confidence[x, y, z] < confidence_threshold:
            continue

        target = repair_target((x, y, z), int(predicted_damage[x, y, z]), current.shape)
        if target is None:
            continue

        tx, ty, tz = target
        if current[tx, ty, tz] == 1:
            continue
        if (
            constrain_to_original
            and original_mask is not None
            and original_mask[tx, ty, tz] != 1
        ):
            continue

        candidates.append((float(confidence[x, y, z]), target))

    return candidates


def apply_repair_candidates(
    current: np.ndarray,
    candidates: List[Tuple[float, float, Tuple[int, int, int]]],
    max_additions: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    candidates.sort(reverse=True)
    if max_additions is not None:
        candidates = candidates[: max(0, max_additions)]

    next_current = current.copy()
    added_mask = np.zeros_like(current, dtype=np.uint8)
    for _, _, target in candidates:
        tx, ty, tz = target
        next_current[tx, ty, tz] = 1
        added_mask[tx, ty, tz] = 1

    return next_current, added_mask, int(added_mask.sum())


def repair_target(source, direction: int, shape) -> Optional[Tuple[int, int, int]]:
    offsets = {
        1: (-1, 0, 0),
        2: (1, 0, 0),
        3: (0, -1, 0),
        4: (0, 1, 0),
        5: (0, 0, -1),
        6: (0, 0, 1),
    }
    if direction not in offsets:
        return None

    target = tuple(source[axis] + offsets[direction][axis] for axis in range(3))
    if any(target[axis] < 0 or target[axis] >= shape[axis] for axis in range(3)):
        return None
    return target
