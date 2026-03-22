from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .dataset import SpectrogramPairDataset
from .ddpm import GaussianDiffusion
from .model import ConditionalUNet
from .pairing import PairingResult, SampleRow, build_paired_rows, subject_sort_key
from .splits import FoldSplit, build_alien_pairs, build_known_subject_folds

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass
class RunSummary:
    lag_name: str
    alien_subjects: str
    fold: int
    train_size: int
    known_test_size: int
    alien_test_size: int
    best_known_test_loss: float
    best_alien_test_loss: float
    final_train_loss: float
    final_known_test_loss: float
    final_alien_test_loss: float


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] | None = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in model.state_dict().items():
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}

    def store(self, model: nn.Module) -> None:
        self.backup = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            raise RuntimeError("EMA backup is empty.")
        model.load_state_dict(self.backup, strict=True)
        self.backup = None


@contextmanager
def ema_scope(model: nn.Module, ema: EMA | None) -> Iterator[None]:
    if ema is None:
        yield
        return
    ema.store(model)
    ema.copy_to(model)
    try:
        yield
    finally:
        ema.restore(model)


def run_experiments(config: ExperimentConfig) -> None:
    set_global_seed(config.rng_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for lag_name, eeg_root in config.lag_roots().items():
        run_lag_experiments(config=config, lag_name=lag_name, eeg_root=eeg_root, device=device)


def run_lag_experiments(
    config: ExperimentConfig,
    lag_name: str,
    eeg_root: str,
    device: torch.device,
) -> None:
    lag_dir = Path(config.output_root) / lag_name
    lag_dir.mkdir(parents=True, exist_ok=True)

    pairing_result = build_paired_rows(config=config, eeg_root=eeg_root)
    if not pairing_result.rows:
        raise ValueError(f"No matched rows were found for {lag_name}.")
    pairing_result.save_summary(lag_dir / "pairing_summary.json")
    save_unmatched_eeg_csv(pairing_result, lag_dir / "unmatched_eeg.csv")

    subjects = sorted({row.subject for row in pairing_result.rows}, key=subject_sort_key)
    alien_pairs, leftovers = build_alien_pairs(subjects, config.alien_pair_stride)
    if not alien_pairs:
        raise ValueError(f"No alien subject pairs could be created for {lag_name}.")

    run_histories: list[tuple[RunSummary, list[dict[str, float]]]] = []
    lag_summary_rows: list[RunSummary] = []

    for alien_pair in alien_pairs:
        alien_pair_dir = lag_dir / f"alien_pair_{'_'.join(alien_pair)}"
        alien_pair_dir.mkdir(parents=True, exist_ok=True)

        alien_rows = [row for row in pairing_result.rows if row.subject in alien_pair]
        known_rows = [row for row in pairing_result.rows if row.subject not in alien_pair]
        fold_splits = build_known_subject_folds(known_rows=known_rows, alien_rows=alien_rows, config=config)

        for fold_split in fold_splits:
            run_dir = alien_pair_dir / f"fold_{fold_split.fold_index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            history, summary = run_single_fold(
                config=config,
                lag_name=lag_name,
                device=device,
                run_dir=run_dir,
                alien_pair=alien_pair,
                fold_split=fold_split,
                leftovers=leftovers,
            )
            run_histories.append((summary, history))
            lag_summary_rows.append(summary)

    aggregate_lag_results(
        config=config,
        lag_dir=lag_dir,
        run_histories=run_histories,
        lag_summary_rows=lag_summary_rows,
    )


def run_single_fold(
    config: ExperimentConfig,
    lag_name: str,
    device: torch.device,
    run_dir: Path,
    alien_pair: tuple[str, ...],
    fold_split: FoldSplit,
    leftovers: list[str],
) -> tuple[list[dict[str, float]], RunSummary]:
    if config.save_json_config:
        config.save_json(run_dir / "config.json")

    subject_payload = {
        "alien_subjects": list(alien_pair),
        "known_subjects": sorted(
            {row.subject for row in (fold_split.train_rows + fold_split.known_test_rows)},
            key=subject_sort_key,
        ),
        "leftover_subjects_without_pair": leftovers,
        "fold_diagnostics": fold_split.diagnostics,
    }
    with (run_dir / "subject_lists.json").open("w", encoding="utf-8") as handle:
        json.dump(subject_payload, handle, indent=2, sort_keys=True)

    if config.save_split_csvs:
        write_rows_csv(run_dir / "split_train.csv", fold_split.train_rows)
        write_rows_csv(run_dir / "split_known_test.csv", fold_split.known_test_rows)
        write_rows_csv(run_dir / "split_alien_test.csv", fold_split.alien_test_rows)

    train_dataset = SpectrogramPairDataset(fold_split.train_rows, config)
    known_dataset = SpectrogramPairDataset(fold_split.known_test_rows, config)
    alien_dataset = SpectrogramPairDataset(fold_split.alien_test_rows, config)

    if train_dataset.shape_info.height != known_dataset.shape_info.height or train_dataset.shape_info.width != known_dataset.shape_info.width:
        raise ValueError("Train and known-test audio shapes do not match.")
    if train_dataset.shape_info.height != alien_dataset.shape_info.height or train_dataset.shape_info.width != alien_dataset.shape_info.width:
        raise ValueError("Train and alien-test audio shapes do not match.")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=True,
        seed=config.rng_seed,
    )
    known_loader = build_dataloader(
        known_dataset,
        batch_size=config.known_eval_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=False,
        seed=config.rng_seed,
    )
    alien_loader = build_dataloader(
        alien_dataset,
        batch_size=config.alien_eval_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=False,
        seed=config.rng_seed,
    )

    model = ConditionalUNet(
        config=config,
        cond_in_ch=train_dataset.shape_info.cond_in_ch,
        target_in_ch=train_dataset.shape_info.target_in_ch,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ema = EMA(model, config.ema_decay) if config.use_ema else None
    diffusion = GaussianDiffusion(
        timesteps=config.ddpm_timesteps,
        cosine_s=config.cosine_s,
    ).to(device)

    history: list[dict[str, float]] = []
    best_known = float("inf")
    best_alien = float("inf")
    best_metric = float("inf")
    amp_enabled = bool(config.use_bf16_amp and device.type == "cuda")

    start_time = time.time()
    for epoch in range(1, config.num_epochs + 1):
        train_one_epoch(
            model=model,
            diffusion=diffusion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
            p_uncond=config.p_uncond,
            amp_enabled=amp_enabled,
            ema=ema,
        )

        with ema_scope(model, ema):
            # These curves intentionally use the DDPM epsilon-prediction objective for train, known-test,
            # and alien-test so every split is measured with the same loss rather than image-space MSE.
            train_loss = evaluate_loader(
                model=model,
                diffusion=diffusion,
                loader=train_loader,
                device=device,
                amp_enabled=amp_enabled,
                seed=config.rng_seed + (epoch * 10) + 1,
            )
            known_loss = evaluate_loader(
                model=model,
                diffusion=diffusion,
                loader=known_loader,
                device=device,
                amp_enabled=amp_enabled,
                seed=config.rng_seed + (epoch * 10) + 2,
            )
            alien_loss = evaluate_loader(
                model=model,
                diffusion=diffusion,
                loader=alien_loader,
                device=device,
                amp_enabled=amp_enabled,
                seed=config.rng_seed + (epoch * 10) + 3,
            )

        elapsed_seconds = time.time() - start_time
        row = {
            "epoch": float(epoch),
            "train_expected_loss": float(train_loss),
            "known_test_loss": float(known_loss),
            "alien_test_loss": float(alien_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": float(elapsed_seconds),
        }
        history.append(row)

        write_curves_csv(run_dir / "curves.csv", history)
        if config.save_plots:
            plot_curves(run_dir / "curves.png", history, title=f"{lag_name} {'/'.join(alien_pair)} fold {fold_split.fold_index}")

        if config.save_last_checkpoint:
            save_checkpoint(
                path=run_dir / "checkpoint_last.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                history=history,
            )

        metric_value = known_loss if config.save_best_by == "known_test_loss" else alien_loss
        if known_loss < best_known:
            best_known = known_loss
            save_checkpoint(
                path=run_dir / "checkpoint_best_known.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                history=history,
            )

        if alien_loss < best_alien:
            best_alien = alien_loss
            if config.save_best_alien_checkpoint:
                save_checkpoint(
                    path=run_dir / "checkpoint_best_alien.pt",
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    history=history,
                )

        metric_value = known_loss if config.save_best_by == "known_test_loss" else alien_loss
        if metric_value < best_metric:
            best_metric = metric_value
            save_checkpoint(
                path=run_dir / "checkpoint_best_selected.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                history=history,
            )

    with ema_scope(model, ema):
        if config.run_generation_eval:
            run_generation_evaluation(
                model=model,
                diffusion=diffusion,
                known_rows=fold_split.known_test_rows,
                alien_rows=fold_split.alien_test_rows,
                config=config,
                device=device,
                run_dir=run_dir,
            )

    final_row = history[-1]
    summary = RunSummary(
        lag_name=lag_name,
        alien_subjects="_".join(alien_pair),
        fold=fold_split.fold_index,
        train_size=len(fold_split.train_rows),
        known_test_size=len(fold_split.known_test_rows),
        alien_test_size=len(fold_split.alien_test_rows),
        best_known_test_loss=float(best_known),
        best_alien_test_loss=float(best_alien),
        final_train_loss=float(final_row["train_expected_loss"]),
        final_known_test_loss=float(final_row["known_test_loss"]),
        final_alien_test_loss=float(final_row["alien_test_loss"]),
    )
    write_summary_txt(run_dir / "summary.txt", summary, subject_payload, fold_split.diagnostics)
    return history, summary


def build_dataloader(
    dataset: SpectrogramPairDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def train_one_epoch(
    model: ConditionalUNet,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    p_uncond: float,
    amp_enabled: bool,
    ema: EMA | None,
) -> None:
    model.train()
    for batch in loader:
        x0 = batch["audio"].to(device, non_blocking=True)
        cond = batch["eeg"].to(device, non_blocking=True)
        timesteps = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = diffusion.q_sample(x0, timesteps, noise)
        uncond_mask = torch.rand(x0.shape[0], device=device) < p_uncond

        optimizer.zero_grad(set_to_none=True)
        with make_autocast(device, amp_enabled):
            eps_hat = model(xt, timesteps, cond, uncond_mask=uncond_mask)
            loss = F.mse_loss(eps_hat.float(), noise.float())
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)


@torch.no_grad()
def evaluate_loader(
    model: ConditionalUNet,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    seed: int,
) -> float:
    model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        x0 = batch["audio"].to(device, non_blocking=True)
        cond = batch["eeg"].to(device, non_blocking=True)
        timesteps = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = diffusion.q_sample(x0, timesteps, noise)
        with make_autocast(device, amp_enabled):
            eps_hat = model(xt, timesteps, cond)
            loss = F.mse_loss(eps_hat.float(), noise.float(), reduction="none")
        loss = loss.mean(dim=(1, 2, 3))
        total_loss += float(loss.sum().item())
        total_samples += int(loss.shape[0])
    if total_samples == 0:
        raise ValueError("Evaluation loader is empty.")
    return total_loss / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: ExperimentConfig,
    history: list[dict[str, float]],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "ema_state": None if ema is None else ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": None,
        "epoch": epoch,
        "config_snapshot": config.to_dict(),
        "loss_history": history,
    }
    torch.save(payload, path)


def run_generation_evaluation(
    model: ConditionalUNet,
    diffusion: GaussianDiffusion,
    known_rows: list[SampleRow],
    alien_rows: list[SampleRow],
    config: ExperimentConfig,
    device: torch.device,
    run_dir: Path,
) -> None:
    guidance_w = config.generation_guidance_w
    if guidance_w is None:
        guidance_w = config.guidance_w_default
    generation_dir = run_dir / "generation_eval"
    generation_dir.mkdir(parents=True, exist_ok=True)

    save_generation_subset(
        rows=known_rows[: config.generation_num_known_samples],
        subset_name="known",
        model=model,
        diffusion=diffusion,
        config=config,
        device=device,
        guidance_w=guidance_w,
        output_dir=generation_dir,
    )
    save_generation_subset(
        rows=alien_rows[: config.generation_num_alien_samples],
        subset_name="alien",
        model=model,
        diffusion=diffusion,
        config=config,
        device=device,
        guidance_w=guidance_w,
        output_dir=generation_dir,
    )


def save_generation_subset(
    rows: list[SampleRow],
    subset_name: str,
    model: ConditionalUNet,
    diffusion: GaussianDiffusion,
    config: ExperimentConfig,
    device: torch.device,
    guidance_w: float,
    output_dir: Path,
) -> None:
    if not rows:
        return
    dataset = SpectrogramPairDataset(rows, config)
    for index in range(len(dataset)):
        sample = dataset[index]
        cond = sample["eeg"].unsqueeze(0).to(device)
        gt = sample["audio"].squeeze(0).numpy()
        generated = diffusion.sample(
            model=model,
            cond=cond,
            output_shape=(1, 1, gt.shape[0], gt.shape[1]),
            guidance_w=guidance_w,
        ).squeeze(0).squeeze(0).cpu().numpy()
        diff = np.abs(generated - gt)
        stem = f"{subset_name}_{index:02d}_{sample['subject']}_{sample['clip_id']}"
        save_array_image(output_dir / f"{stem}_generated.png", generated)
        save_array_image(output_dir / f"{stem}_gt.png", gt)
        save_array_image(output_dir / f"{stem}_diff.png", diff)


def save_array_image(path: Path, array: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(array, aspect="auto", origin="lower", cmap="magma")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def aggregate_lag_results(
    config: ExperimentConfig,
    lag_dir: Path,
    run_histories: list[tuple[RunSummary, list[dict[str, float]]]],
    lag_summary_rows: list[RunSummary],
) -> None:
    aggregate_dir = lag_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    epoch_buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for _, history in run_histories:
        for row in history:
            epoch = int(row["epoch"])
            epoch_buckets[epoch]["train_expected_loss"].append(float(row["train_expected_loss"]))
            epoch_buckets[epoch]["known_test_loss"].append(float(row["known_test_loss"]))
            epoch_buckets[epoch]["alien_test_loss"].append(float(row["alien_test_loss"]))

    aggregate_rows: list[dict[str, float]] = []
    for epoch in sorted(epoch_buckets):
        bucket = epoch_buckets[epoch]
        aggregate_rows.append(
            {
                "epoch": float(epoch),
                "train_mean": float(np.mean(bucket["train_expected_loss"])),
                "train_std": float(np.std(bucket["train_expected_loss"])),
                "known_mean": float(np.mean(bucket["known_test_loss"])),
                "known_std": float(np.std(bucket["known_test_loss"])),
                "alien_mean": float(np.mean(bucket["alien_test_loss"])),
                "alien_std": float(np.std(bucket["alien_test_loss"])),
            }
        )

    write_dict_csv(aggregate_dir / "aggregate_curves.csv", aggregate_rows)
    if config.save_plots:
        plot_aggregate_curves(aggregate_dir / "aggregate_curves.png", aggregate_rows, title=lag_dir.name)
    write_dict_csv(
        aggregate_dir / "lag_summary.csv",
        [asdict(row) for row in lag_summary_rows],
    )


def plot_curves(path: Path, history: list[dict[str, float]], title: str) -> None:
    epochs = [int(row["epoch"]) for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_expected_loss"] for row in history], label="train_expected_loss")
    ax.plot(epochs, [row["known_test_loss"] for row in history], label="known_test_loss")
    ax.plot(epochs, [row["alien_test_loss"] for row in history], label="alien_test_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epsilon-prediction MSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_aggregate_curves(path: Path, rows: list[dict[str, float]], title: str) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_mean"] for row in rows], label="train_mean")
    ax.plot(epochs, [row["known_mean"] for row in rows], label="known_mean")
    ax.plot(epochs, [row["alien_mean"] for row in rows], label="alien_mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean epsilon-prediction MSE")
    ax.set_title(f"Aggregate curves: {title}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_curves_csv(path: Path, history: list[dict[str, float]]) -> None:
    write_dict_csv(path, history)


def write_rows_csv(path: Path, rows: list[SampleRow]) -> None:
    payload = [
        {
            "subject": row.subject,
            "word": row.word,
            "stimulus": row.stimulus,
            "interval_index": row.interval_index,
            "clip_id": row.clip_id,
            "audio_npy_path": row.audio_npy_path,
            "eeg_npy_path": row.eeg_npy_path,
            "filename_stem": row.filename_stem,
        }
        for row in rows
    ]
    write_dict_csv(path, payload)


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_txt(
    path: Path,
    summary: RunSummary,
    subject_payload: dict[str, object],
    diagnostics: list[str],
) -> None:
    lines = [
        f"lag_name: {summary.lag_name}",
        f"alien_subjects: {summary.alien_subjects}",
        f"fold: {summary.fold}",
        f"train_size: {summary.train_size}",
        f"known_test_size: {summary.known_test_size}",
        f"alien_test_size: {summary.alien_test_size}",
        f"best_known_test_loss: {summary.best_known_test_loss:.6f}",
        f"best_alien_test_loss: {summary.best_alien_test_loss:.6f}",
        f"final_train_loss: {summary.final_train_loss:.6f}",
        f"final_known_test_loss: {summary.final_known_test_loss:.6f}",
        f"final_alien_test_loss: {summary.final_alien_test_loss:.6f}",
        "",
        "Subject split notes:",
        json.dumps(subject_payload, indent=2, sort_keys=True),
        "",
        "Rolling fold diagnostics:",
        "\n".join(diagnostics) if diagnostics else "None",
        "",
        "Loss-curve note:",
        "The three per-epoch curves are all epsilon-prediction MSE values on DDPM noise targets.",
        "This keeps train, known-test, and alien-test directly comparable and avoids mixing in separate image-generation metrics.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_unmatched_eeg_csv(pairing_result: PairingResult, path: Path) -> None:
    rows = [{"eeg_npy_path": value} for value in pairing_result.summary.unmatched_eeg_paths]
    write_dict_csv(path, rows)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_autocast(device: torch.device, amp_enabled: bool):
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled)
    return nullcontext()
