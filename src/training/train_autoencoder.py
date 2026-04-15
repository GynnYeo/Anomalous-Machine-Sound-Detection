"""High-level baseline training orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.conv_autoencoder import ConvAutoencoder
from src.data.dataset import MIMIIDataset
from src.data.transforms import BaselineLogMelTransform
from src.training.autoencoder_trainer import (
    train_autoencoder_one_epoch,
    validate_autoencoder_one_epoch,
)

from src.training.callbacks import (
    load_checkpoint,
    load_history,
    save_checkpoint,
    save_history,
    save_json,
)

from src.training.losses import build_baseline_loss
from src.training.trainer import train_one_epoch, validate_one_epoch
from src.utils.io import to_portable_path
from src.utils.seed import set_seed


def select_device() -> torch.device:
    """Select the best available device with CUDA first, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_autoencoder_training(
    manifest_path: str | Path,
    epochs: int,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    seed: int = 42,
    run_name: str = "autoencoder",
    checkpoint_dir: str | Path = "artifacts/checkpoints",
    history_dir: str | Path = "artifacts/metrics",
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run the baseline training loop and save artifacts under run-specific folders."""
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")

    set_seed(seed)
    device = select_device()
    transform = BaselineLogMelTransform()

    train_dataset = MIMIIDataset(
        manifest_path=manifest_path,
        split="train",
        transform=transform,
        return_metadata=False,
        label_filter="normal",
    )
    val_dataset = MIMIIDataset(
        manifest_path=manifest_path,
        split="val",
        transform=transform,
        return_metadata=False,
    )

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )

    model = ConvAutoencoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    history_root = Path(history_dir).expanduser().resolve()
    checkpoint_run_dir = checkpoint_root / run_name
    history_run_dir = history_root / run_name
    best_checkpoint_path = checkpoint_run_dir / "best.pt"
    last_checkpoint_path = checkpoint_run_dir / "last.pt"
    history_path = history_run_dir / "history.json"
    run_config_path = history_run_dir / "run_config.json"

    history = _build_initial_history(
        run_name=run_name,
        manifest_path=manifest_path,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    start_epoch = 1
    best_val_loss = float("inf")

    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])

        if history_path.exists():
            history = load_history(history_path)
        print(f"Resumed training from checkpoint: {Path(resume_from).expanduser().resolve()}")

    history["best_checkpoint_path"] = to_portable_path(best_checkpoint_path)
    history["last_checkpoint_path"] = to_portable_path(last_checkpoint_path)
    history["history_path"] = to_portable_path(history_path)
    history["run_config_path"] = to_portable_path(run_config_path)

    run_config = _build_run_config(
        run_name=run_name,
        manifest_path=manifest_path,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        epochs_requested=epochs,
        num_workers=num_workers,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        transform=transform,
        checkpoint_dir=checkpoint_run_dir,
        history_dir=history_run_dir,
    )
    save_json(run_config, run_config_path)

    print(
        f"Starting training on {device.type} | "
        f"train_samples={len(train_dataset)} | val_samples={len(val_dataset)}"
    )

    previous_total_training_time = history.get("total_training_time_seconds")
    if previous_total_training_time is None:
        previous_total_training_time = 0.0

    total_start_time = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.perf_counter()

        train_metrics = train_autoencoder_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate_autoencoder_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time_seconds = time.perf_counter() - epoch_start_time
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "epoch_time_seconds": float(epoch_time_seconds),
        }
        history["epochs"].append(epoch_record)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "run_name": run_name,
            "manifest_path": to_portable_path(manifest_path),
        }

        improved = epoch_record["val_loss"] < best_val_loss
        if improved:
            best_val_loss = epoch_record["val_loss"]
            history["best_epoch"] = epoch
            history["best_val_loss"] = best_val_loss
            checkpoint["best_val_loss"] = best_val_loss
            save_checkpoint(checkpoint, best_checkpoint_path)

        checkpoint["best_val_loss"] = best_val_loss
        save_checkpoint(checkpoint, last_checkpoint_path)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={epoch_record['train_loss']:.4f} | "
            f"val_loss={epoch_record['val_loss']:.4f} | "
            f"time={epoch_record['epoch_time_seconds']:.2f}s"
        )

        history["completed_epochs"] = len(history["epochs"])
        save_history(history, history_path)

    total_training_time_seconds = previous_total_training_time + (
        time.perf_counter() - total_start_time
    )
    history["total_training_time_seconds"] = float(total_training_time_seconds)
    history["completed_epochs"] = len(history["epochs"])
    save_history(history, history_path)

    print(
        f"Training complete | total_time={total_training_time_seconds:.2f}s | "
        f"best_val_loss={history.get('best_val_loss', float('nan')):.4f}"
    )
    return history


def _build_initial_history(
    run_name: str,
    manifest_path: str | Path,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Create the initial history structure for a training run."""
    return {
        "run_name": run_name,
        "manifest_path": to_portable_path(manifest_path),
        "device": device.type,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "epochs": [],
        "completed_epochs": 0,
        "best_epoch": None,
        "best_val_loss": None,
        "total_training_time_seconds": None,
    }


def _build_run_config(
    run_name: str,
    manifest_path: str | Path,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    seed: int,
    epochs_requested: int,
    num_workers: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    transform: BaselineLogMelTransform,
    checkpoint_dir: Path,
    history_dir: Path,
) -> dict[str, Any]:
    """Build a small JSON-serializable run configuration record."""
    return {
        "run_name": run_name,
        "manifest_path": to_portable_path(manifest_path),
        "device": device.type,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "epochs_requested": epochs_requested,
        "num_workers": num_workers,
        "model_name": model.__class__.__name__,
        "loss_name": criterion.__class__.__name__,
        "optimizer_name": optimizer.__class__.__name__,
        "preprocessing_name": transform.__class__.__name__,
        "preprocessing_parameters": {
            "sample_rate": transform.sample_rate,
            "n_fft": transform.n_fft,
            "win_length": transform.win_length,
            "hop_length": transform.hop_length,
            "n_mels": transform.n_mels,
            "f_min": transform.f_min,
            "f_max": transform.f_max,
        },
        "checkpoint_dir": to_portable_path(checkpoint_dir),
        "history_dir": to_portable_path(history_dir),
    }
