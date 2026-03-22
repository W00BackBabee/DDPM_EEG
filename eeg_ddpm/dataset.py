from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .pairing import SampleRow


@dataclass
class DatasetShapeInfo:
    cond_in_ch: int
    target_in_ch: int
    height: int
    width: int


class SpectrogramPairDataset(Dataset):
    def __init__(self, rows: list[SampleRow], config: ExperimentConfig):
        if not rows:
            raise ValueError("Dataset received zero rows.")
        self.rows = rows
        self.config = config
        self.shape_info = self._infer_shape_info()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        audio = self._load_audio(row.audio_npy_path)
        eeg = self._load_eeg(row.eeg_npy_path)
        return {
            "audio": torch.from_numpy(audio),
            "eeg": torch.from_numpy(eeg),
            "subject": row.subject,
            "word": row.word,
            "clip_id": row.clip_id,
            "stimulus": row.stimulus,
            "interval_index": row.interval_index,
            "audio_npy_path": row.audio_npy_path,
            "eeg_npy_path": row.eeg_npy_path,
        }

    def _infer_shape_info(self) -> DatasetShapeInfo:
        first_row = self.rows[0]
        eeg = self._load_eeg(first_row.eeg_npy_path)
        audio = self._load_audio(first_row.audio_npy_path)
        return DatasetShapeInfo(
            cond_in_ch=int(eeg.shape[0]),
            target_in_ch=int(audio.shape[0]),
            height=int(audio.shape[-2]),
            width=int(audio.shape[-1]),
        )

    def _load_audio(self, path: str) -> np.ndarray:
        array = np.load(path).astype(np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = np.moveaxis(array, -1, 0)
        elif array.ndim != 3:
            raise ValueError(f"Audio spectrogram must have 2 or 3 dims, got shape={array.shape} for {path}")

        if array.shape[0] != 1:
            raise ValueError(
                f"Audio spectrogram is expected to have one channel, got shape={array.shape} for {path}"
            )
        return self._rescale(array)

    def _load_eeg(self, path: str) -> np.ndarray:
        array = np.load(path).astype(np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        elif array.ndim != 3:
            raise ValueError(f"EEG spectrogram must have 2 or 3 dims, got shape={array.shape} for {path}")

        if self.config.eeg_channel_indices:
            max_index = array.shape[0] - 1
            invalid = [index for index in self.config.eeg_channel_indices if index < 0 or index > max_index]
            if invalid:
                raise IndexError(
                    f"EEG channel index out of bounds for {path}. "
                    f"Requested={invalid}, available=[0, {max_index}]"
                )
            array = array[self.config.eeg_channel_indices, ...]
        return self._rescale(array)

    def _rescale(self, array: np.ndarray) -> np.ndarray:
        if self.config.spectrogram_scale_mode == "none":
            return array
        if self.config.spectrogram_scale_max <= self.config.spectrogram_scale_min:
            raise ValueError("spectrogram_scale_max must be greater than spectrogram_scale_min.")
        scaled = (array - self.config.spectrogram_scale_min) / (
            self.config.spectrogram_scale_max - self.config.spectrogram_scale_min
        )
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 2.0) - 1.0


def describe_dataset(rows: list[SampleRow], config: ExperimentConfig) -> DatasetShapeInfo:
    return SpectrogramPairDataset(rows, config).shape_info
