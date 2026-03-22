from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .config import ExperimentConfig
from .pairing import SampleRow, row_sort_key, subject_sort_key


@dataclass
class FoldSplit:
    fold_index: int
    train_rows: list[SampleRow]
    known_test_rows: list[SampleRow]
    alien_test_rows: list[SampleRow]
    diagnostics: list[str]


def build_alien_pairs(subjects: list[str], stride: int) -> tuple[list[tuple[str, ...]], list[str]]:
    ordered = sorted(subjects, key=subject_sort_key)
    pairs: list[tuple[str, ...]] = []
    leftovers: list[str] = []
    cursor = len(ordered)
    while cursor > 0:
        start = max(0, cursor - stride)
        chunk = ordered[start:cursor]
        # Alien groups are created from the highest subject IDs downward so held-out subjects come from the tail.
        if len(chunk) == stride:
            pairs.append(tuple(chunk))
        else:
            leftovers.extend(chunk)
        cursor = start
    return pairs, leftovers


def build_known_subject_folds(
    known_rows: list[SampleRow],
    alien_rows: list[SampleRow],
    config: ExperimentConfig,
) -> list[FoldSplit]:
    if not known_rows:
        raise ValueError("Known-subject pool is empty; cannot build training/test folds.")
    if not alien_rows:
        raise ValueError("Alien-subject pool is empty; cannot build alien evaluation set.")

    grouped: dict[tuple[str, str], list[SampleRow]] = defaultdict(list)
    for row in known_rows:
        grouped[(row.subject, row.word)].append(row)

    fold_tests: dict[int, list[SampleRow]] = {fold: [] for fold in range(1, config.rolling_word_folds + 1)}
    diagnostics: dict[int, list[str]] = {fold: [] for fold in range(1, config.rolling_word_folds + 1)}
    max_available_folds = 0

    for (subject, word), rows in sorted(grouped.items()):
        ordered_rows = sorted(
            rows,
            key=lambda row: (row.interval_index, row.clip_id, row.filename_stem),
        )
        n = len(ordered_rows)
        block_size = math.floor(config.rolling_word_test_fraction * n)
        if block_size < 1 and n > 0:
            block_size = 1

        if n == 0 or block_size == 0:
            continue

        available_folds = min(config.rolling_word_folds, n // block_size)
        max_available_folds = max(max_available_folds, available_folds)
        achieved_coverage = (available_folds * block_size) / max(n, 1)
        if available_folds < config.rolling_word_folds:
            message = (
                f"{subject}/{word}: only {available_folds} disjoint fold(s) available for n={n}, "
                f"block_size={block_size}."
            )
            for fold in range(available_folds + 1, config.rolling_word_folds + 1):
                diagnostics[fold].append(message)
        if achieved_coverage < config.rolling_word_coverage_target:
            coverage_message = (
                f"{subject}/{word}: achieved coverage {achieved_coverage:.2f} "
                f"is below target {config.rolling_word_coverage_target:.2f}."
            )
            for fold in range(1, available_folds + 1):
                diagnostics[fold].append(coverage_message)

        for fold_zero_based in range(available_folds):
            fold_index = fold_zero_based + 1
            # Each fold takes a non-overlapping 30% block from the end of the ordered subject-word sequence.
            end = n - (fold_zero_based * block_size)
            start = max(0, end - block_size)
            fold_tests[fold_index].extend(ordered_rows[start:end])

    if max_available_folds == 0:
        raise ValueError(
            "No rolling folds could be built. Check that known-subject rows exist after filtering."
        )

    all_known_rows = sorted(known_rows, key=row_sort_key)
    fold_splits: list[FoldSplit] = []
    for fold_index in range(1, max_available_folds + 1):
        test_ids = {row_identity(row) for row in fold_tests[fold_index]}
        known_test_rows = sorted(fold_tests[fold_index], key=row_sort_key)
        train_rows = sorted(
            [row for row in all_known_rows if row_identity(row) not in test_ids],
            key=row_sort_key,
        )
        if not train_rows:
            raise ValueError(f"Fold {fold_index} has an empty training split.")
        if not known_test_rows:
            raise ValueError(f"Fold {fold_index} has an empty known-subject test split.")
        fold_splits.append(
            FoldSplit(
                fold_index=fold_index,
                train_rows=train_rows,
                known_test_rows=known_test_rows,
                alien_test_rows=sorted(alien_rows, key=row_sort_key),
                diagnostics=sorted(set(diagnostics[fold_index])),
            )
        )
    return fold_splits


def row_identity(row: SampleRow) -> tuple[str, str, int, int, str]:
    return row.subject, row.word, row.stimulus, row.interval_index, row.eeg_npy_path
