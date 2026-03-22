from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ExperimentConfig

FILENAME_PATTERN = re.compile(
    r"SoundFile(?P<stimulus>\d+)[_-](?P<interval>\d+)[_-](?P<word>.+)$",
    re.IGNORECASE,
)
SUBJECT_PATTERN = re.compile(r"(S\d{2,3})", re.IGNORECASE)


@dataclass(frozen=True)
class SampleRow:
    subject: str
    word: str
    stimulus: int
    interval_index: int
    clip_id: str
    audio_npy_path: str
    eeg_npy_path: str
    filename_stem: str


@dataclass
class PairingSummary:
    audio_file_count: int
    eeg_file_count: int
    total_before_filtering: int
    total_after_filtering: int
    unmatched_eeg_count: int
    unmatched_eeg_paths: list[str]
    unique_subjects: list[str]
    unique_words: list[str]
    per_word_counts_after_filtering: dict[str, int]


@dataclass
class PairingResult:
    rows: list[SampleRow]
    summary: PairingSummary

    def save_summary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.summary), handle, indent=2, sort_keys=True)


def normalize_word(word: str) -> str:
    return word.strip().lower()


def canonical_clip_id(stimulus: int, interval_index: int, word: str) -> str:
    return f"SoundFile{stimulus}_{interval_index:04d}_{normalize_word(word)}"


def parse_clip_from_path(path: str | Path) -> tuple[int, int, str, str]:
    path = Path(path)
    stem = path.stem.strip()
    # Filenames are matched directly from the shared spectrogram stem so no manifest is needed.
    match = FILENAME_PATTERN.search(stem)
    if match is None:
        raise ValueError(
            f"Could not parse filename '{path.name}'. Expected a stem containing "
            "SoundFile<stimulus>_<interval_index>_<word>."
        )
    stimulus = int(match.group("stimulus"))
    interval_index = int(match.group("interval"))
    word = normalize_word(match.group("word"))
    return stimulus, interval_index, word, canonical_clip_id(stimulus, interval_index, word)


def infer_subject(path: str | Path) -> str:
    path = Path(path)
    search_fields = list(path.parts) + [path.stem]
    for field in reversed(search_fields):
        match = SUBJECT_PATTERN.search(field)
        if match:
            digits = int(re.sub(r"\D", "", match.group(1)))
            return f"S{digits:02d}"
    raise ValueError(
        f"Could not infer subject ID from '{path}'. Expected a directory or filename containing S##."
    )


def _iter_npy_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Missing root directory: {root_path}")
    return sorted(path for path in root_path.rglob("*.npy") if path.is_file())


def build_paired_rows(config: ExperimentConfig, eeg_root: str | Path) -> PairingResult:
    audio_files = _iter_npy_files(config.audio_root)
    eeg_files = _iter_npy_files(eeg_root)

    audio_map: dict[tuple[int, int, str], str] = {}
    for audio_path in audio_files:
        stimulus, interval_index, word, _ = parse_clip_from_path(audio_path)
        key = (stimulus, interval_index, word)
        if key in audio_map:
            raise ValueError(
                f"Duplicate audio key {key}. Existing='{audio_map[key]}', duplicate='{audio_path}'."
            )
        audio_map[key] = str(audio_path.resolve())

    eeg_map: dict[tuple[str, int, int, str], str] = {}
    unmatched_eeg_paths: list[str] = []
    rows: list[SampleRow] = []

    for eeg_path in eeg_files:
        subject = infer_subject(eeg_path)
        stimulus, interval_index, word, clip_id = parse_clip_from_path(eeg_path)
        eeg_key = (subject, stimulus, interval_index, word)
        if eeg_key in eeg_map:
            raise ValueError(
                f"Duplicate EEG key {eeg_key}. Existing='{eeg_map[eeg_key]}', duplicate='{eeg_path}'."
            )
        eeg_map[eeg_key] = str(eeg_path.resolve())

        audio_key = (stimulus, interval_index, word)
        audio_path = audio_map.get(audio_key)
        if audio_path is None:
            unmatched_eeg_paths.append(str(eeg_path.resolve()))
            continue

        # Final rows are many-to-one by design: multiple subject-specific EEG files can share one audio target.
        rows.append(
            SampleRow(
                subject=subject,
                word=word,
                stimulus=stimulus,
                interval_index=interval_index,
                clip_id=clip_id,
                audio_npy_path=audio_path,
                eeg_npy_path=str(eeg_path.resolve()),
                filename_stem=eeg_path.stem,
            )
        )

    total_before_filtering = len(rows)
    rows = apply_word_filters(rows, config.include_words, config.exclude_words)
    rows = sorted(rows, key=row_sort_key)
    per_word_counts = Counter(row.word for row in rows)

    summary = PairingSummary(
        audio_file_count=len(audio_files),
        eeg_file_count=len(eeg_files),
        total_before_filtering=total_before_filtering,
        total_after_filtering=len(rows),
        unmatched_eeg_count=len(unmatched_eeg_paths),
        unmatched_eeg_paths=unmatched_eeg_paths,
        unique_subjects=sorted({row.subject for row in rows}, key=subject_sort_key),
        unique_words=sorted(per_word_counts),
        per_word_counts_after_filtering=dict(sorted(per_word_counts.items())),
    )
    return PairingResult(rows=rows, summary=summary)


def apply_word_filters(
    rows: list[SampleRow],
    include_words: list[str] | None,
    exclude_words: list[str] | None,
) -> list[SampleRow]:
    include = set(include_words or [])
    exclude = set(exclude_words or [])
    filtered: list[SampleRow] = []
    for row in rows:
        if include and row.word not in include:
            continue
        if row.word in exclude:
            continue
        filtered.append(row)
    return filtered


def row_sort_key(row: SampleRow) -> tuple[int, int, str, int, str, str]:
    return (
        int(re.sub(r"\D", "", row.subject)),
        row.stimulus,
        row.interval_index,
        row.word,
        row.clip_id,
        row.filename_stem,
    )


def subject_sort_key(subject: str) -> tuple[int, str]:
    digits = int(re.sub(r"\D", "", subject))
    return digits, subject
