# DDPM_EEG

EEG-conditioned denoising diffusion training code for predicting audio spectrograms from precomputed EEG spectrogram inputs.

## Repository layout

- `main.py`: command-line entrypoint
- `eeg_ddpm/`: public Python package and training logic
- `configs/example_config.json`: example experiment configuration

## Requirements

Install the core dependencies:

```bash
pip install -r requirements.txt
```

## Run

Update `configs/example_config.json` with your dataset paths, then run:

```bash
python main.py --config configs/example_config.json
```

The training script expects precomputed `.npy` spectrogram files for:

- audio targets
- EEG spectrograms for each lag directory

Outputs are written under the configured `output_root`.
