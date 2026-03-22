"""Public package exports for the EEG-conditioned DDPM project."""

from .config import ExperimentConfig, load_config
from .train import run_experiments

__all__ = ["ExperimentConfig", "load_config", "run_experiments"]
