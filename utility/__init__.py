"""EEG-conditioned DDPM training package."""

from .config import ExperimentConfig, load_config


def run_experiments(*args, **kwargs):
    from .train import run_experiments as _run_experiments

    return _run_experiments(*args, **kwargs)


__all__ = ["ExperimentConfig", "load_config", "run_experiments"]
