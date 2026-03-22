from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    # Paths
    audio_root: str = "/path/to/audio_spectrograms"
    eeg_root_0ms: str = "/path/to/eeg_spectrograms/lag_0ms"
    eeg_root_100ms: str = "/path/to/eeg_spectrograms/lag_100ms"
    eeg_root_300ms: str = "/path/to/eeg_spectrograms/lag_300ms"
    output_root: str = "./outputs"

    # Word filtering
    include_words: list[str] | None = None
    exclude_words: list[str] | None = None

    # EEG channel selection
    eeg_channel_indices: list[int] | None = None

    # Data scaling
    spectrogram_scale_mode: str = "none"  # "none" or "minmax_neg1_1"
    spectrogram_scale_min: float = 0.0
    spectrogram_scale_max: float = 1.0

    # Training
    rng_seed: int = 1337
    num_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    grad_clip_norm: float = 1.0
    ddpm_timesteps: int = 1000
    use_bf16_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    p_uncond: float = 0.2
    guidance_w_default: float = 1.5

    # Model
    base_ch: int = 32
    ch_mults: tuple[int, ...] = (1, 2, 4, 8)
    cond_ch: int = 128
    inj_ch: int = 16
    cond_scale: float = 1.5
    emb_dim: int = 256

    # Schedule
    beta_schedule: str = "cosine"
    cosine_s: float = 0.008

    # Evaluation
    eval_every_epoch: bool = True
    known_eval_batch_size: int = 8
    alien_eval_batch_size: int = 8
    save_best_by: str = "known_test_loss"
    save_last_checkpoint: bool = True
    save_best_alien_checkpoint: bool = True

    # Cross-validation
    alien_pair_stride: int = 2
    rolling_word_test_fraction: float = 0.30
    rolling_word_folds: int = 3
    rolling_word_coverage_target: float = 0.90

    # Logging
    save_plots: bool = True
    save_json_config: bool = True
    save_split_csvs: bool = True

    # Optional generation evaluation
    run_generation_eval: bool = False
    generation_num_known_samples: int = 4
    generation_num_alien_samples: int = 4
    generation_guidance_w: float | None = None

    def __post_init__(self) -> None:
        self.include_words = _normalize_word_list(self.include_words)
        self.exclude_words = _normalize_word_list(self.exclude_words)
        if isinstance(self.ch_mults, list):
            self.ch_mults = tuple(int(v) for v in self.ch_mults)
        if self.eeg_channel_indices is not None and len(self.eeg_channel_indices) == 0:
            self.eeg_channel_indices = None
        if self.include_words and self.exclude_words:
            overlap = sorted(set(self.include_words) & set(self.exclude_words))
            if overlap:
                raise ValueError(f"Words cannot be both included and excluded: {overlap}")
        if self.alien_pair_stride < 2:
            raise ValueError("alien_pair_stride must be at least 2.")
        if self.rolling_word_folds < 1:
            raise ValueError("rolling_word_folds must be at least 1.")
        if not (0.0 < self.rolling_word_test_fraction < 1.0):
            raise ValueError("rolling_word_test_fraction must be between 0 and 1.")
        if self.spectrogram_scale_mode not in {"none", "minmax_neg1_1"}:
            raise ValueError("spectrogram_scale_mode must be 'none' or 'minmax_neg1_1'.")
        if self.save_best_by not in {"known_test_loss", "alien_test_loss"}:
            raise ValueError("save_best_by must be 'known_test_loss' or 'alien_test_loss'.")
        if self.beta_schedule != "cosine":
            raise ValueError("Only cosine beta schedule is implemented.")

    def lag_roots(self) -> dict[str, str]:
        return {
            "lag_0ms": self.eeg_root_0ms,
            "lag_100ms": self.eeg_root_100ms,
            "lag_300ms": self.eeg_root_300ms,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


def _normalize_word_list(words: list[str] | None) -> list[str] | None:
    if words is None:
        return None
    normalized = [str(word).strip().lower() for word in words if str(word).strip()]
    return normalized or None


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    valid_keys = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")
    return ExperimentConfig(**raw)
