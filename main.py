from __future__ import annotations

import argparse

from eeg_ddpm import load_config, run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an EEG-conditioned DDPM using precomputed spectrogram .npy files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example_config.json",
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_experiments(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
