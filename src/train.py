"""Two-Stage Transfer Learning for Tuberculosis (TB) Chest X-ray Classification.

Key Features:
- Stage 1: Freeze backbone, train classification head (~2.5k params)
- Stage 2: Unfreeze top convolutional blocks, fine-tune with lower LR (~1.1M params)
- Class-weighted CrossEntropyLoss to address the 5:1 Normal:TB imbalance
- AdamW optimizer + Cosine Annealing learning rate schedule
- Sensitivity/Recall-driven checkpointing (best model prioritizes detecting TB)
- Early stopping mechanism
- Generation of training curves (Loss, Accuracy, TB Recall) saved to results/training_curves.png
- Portable: Runs on CPU, Apple MPS, or CUDA GPU; supports Google Colab
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    CLASS_NAMES,
    create_stratified_splits,
    find_dataset_images,
    get_dataloaders,
    get_device,
)
from src.model import TBClassifier, get_model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train the model for one full epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: PyTorch optimizer.
        device: Target compute device.

    Returns:
        Tuple of (epoch_loss, epoch_accuracy, epoch_tb_recall).
    """
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    total_samples = len(all_targets)
    epoch_loss = running_loss / total_samples
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_targets))
    # Recall for TB (class 1)
    epoch_tb_recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0.0)

    return float(epoch_loss), float(epoch_acc), float(epoch_tb_recall)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    """Evaluate model performance on validation or test dataset.

    Args:
        model: PyTorch model.
        dataloader: Evaluation DataLoader.
        criterion: Loss function.
        device: Target compute device.

    Returns:
        Tuple of (loss, accuracy, tb_recall, tb_precision, tb_f1).
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    total_samples = len(all_targets)
    loss = running_loss / total_samples
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    tb_recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0.0)
    
    # Specificity is recall for Normal class (pos_label=0)
    specificity = recall_score(all_targets, all_preds, pos_label=0, zero_division=0.0)
    tb_f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0.0)

    return float(loss), float(acc), float(tb_recall), float(specificity), float(tb_f1)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = "results/training_curves.png",
) -> None:
    """Generate and save publication-grade training progression curves.

    Args:
        history: Dictionary containing recorded metrics per epoch.
        save_path: Destination path for the plot.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss Curve
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Accuracy Curve
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc", color="#2ca02c", lw=2)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc", color="#d62728", lw=2)
    axes[1].set_title("Overall Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. Clinical TB Sensitivity / Recall Curve (Most critical for screening)
    axes[2].plot(epochs, [r * 100 for r in history["train_tb_recall"]], "o-", label="Train TB Recall", color="#9467bd", lw=2)
    axes[2].plot(epochs, [r * 100 for r in history["val_tb_recall"]], "s--", label="Val TB Recall (Sensitivity)", color="#e377c2", lw=2)
    axes[2].axhline(y=95.0, color="gray", linestyle=":", label="WHO Target (95%)")
    axes[2].set_title("Clinical TB Sensitivity / Recall (%)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Sensitivity (%)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle("TB-Screen AI: Two-Stage Transfer Learning Progress (EfficientNet-B0)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def train_model(
    data_dir: str = "data/raw",
    output_dir: str = "models",
    results_dir: str = "results",
    stage1_epochs: int = 3,
    stage2_epochs: int = 7,
    batch_size: int = 32,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    seed: int = 42,
    patience: int = 4,
) -> Dict[str, any]:
    """Execute complete two-stage transfer learning pipeline.

    Args:
        data_dir: Path to directory containing raw images.
        output_dir: Directory where model weights are stored.
        results_dir: Directory where training curves and logs are stored.
        stage1_epochs: Epochs for Stage 1 head warmup.
        stage2_epochs: Epochs for Stage 2 fine-tuning.
        batch_size: DataLoader batch size.
        lr_stage1: Head learning rate.
        lr_stage2: Fine-tuning learning rate.
        seed: Random seed.
        patience: Early stopping patience epochs.

    Returns:
        Summary dictionary with training results.
    """
    device = get_device()
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Active Hardware'})")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Prepare data splits and loaders
    manifest_file = os.path.join("data/processed", "splits_manifest.json")
    if os.path.exists(manifest_file):
        print(f"[*] Loading stratified manifest from: {manifest_file}")
        with open(manifest_file, "r") as f:
            splits = json.load(f)
    else:
        print("[*] Generating new stratified dataset splits...")
        image_paths, labels = find_dataset_images(data_dir)
        splits = create_stratified_splits(image_paths, labels, seed=seed, save_dir="data/processed")

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(splits, batch_size=batch_size)
    print(f"[*] Dataset: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    print(f"[*] Class Weights for Loss: Normal={class_weights[0]:.4f}, TB={class_weights[1]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 2. Instantiate Model
    model = get_model(pretrained=True, num_classes=2).to(device)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_tb_recall": [], "val_tb_recall": [],
        "val_specificity": [], "val_tb_f1": [],
    }

    best_val_tb_recall = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    total_epochs = stage1_epochs + stage2_epochs
    global_epoch = 0

    # ==========================================
    # STAGE 1: HEAD WARMUP (BACKBONE FROZEN)
    # ==========================================
    print("\n" + "="*60)
    print("STAGE 1: Classification Head Warmup (Backbone Frozen)")
    print("="*60)
    model.freeze_backbone()
    total_p, trainable_p = model.count_parameters()
    print(f"[*] Total Params: {total_p:,} | Trainable Params: {trainable_p:,}")

    optimizer_s1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_stage1,
        weight_decay=1e-4,
    )
    scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=stage1_epochs, eta_min=1e-5)

    for epoch in range(1, stage1_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc, tr_rec = train_one_epoch(model, train_loader, criterion, optimizer_s1, device)
        scheduler_s1.step()
        val_loss, val_acc, val_rec, val_spec, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_tb_recall"].append(tr_rec)
        history["val_tb_recall"].append(val_rec)
        history["val_specificity"].append(val_spec)
        history["val_tb_f1"].append(val_f1)

        print(
            f"Epoch [{global_epoch:02d}/{total_epochs:02d}] (S1) - "
            f"Loss: {tr_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {tr_acc*100:.1f}%/{val_acc*100:.1f}% | "
            f"TB Recall: {tr_rec*100:.1f}%/{val_rec*100:.1f}% | "
            f"F1: {val_f1:.4f} | {elapsed:.1f}s"
        )

        # Checkpoint based on Validation TB Recall, breaking ties with val_loss
        if (val_rec > best_val_tb_recall) or (val_rec == best_val_tb_recall and val_loss < best_val_loss):
            best_val_tb_recall = val_rec
            best_val_loss = val_loss
            best_epoch = global_epoch
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_tb_recall": best_val_tb_recall,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_specificity": val_spec,
                    "class_names": CLASS_NAMES,
                    "architecture": "efficientnet_b0",
                    "class_weights": class_weights.tolist(),
                },
                best_model_path,
            )
            print(f"  --> Saved new best checkpoint: TB Recall={best_val_tb_recall*100:.1f}%")

    # ==========================================
    # STAGE 2: FINE-TUNING TOP BLOCKS
    # ==========================================
    print("\n" + "="*60)
    print("STAGE 2: Fine-Tuning Top Convolutional Blocks")
    print("="*60)
    model.unfreeze_blocks(num_blocks=2)
    total_p, trainable_p = model.count_parameters()
    print(f"[*] Total Params: {total_p:,} | Trainable Params: {trainable_p:,}")

    optimizer_s2 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_stage2,
        weight_decay=1e-4,
    )
    scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=stage2_epochs, eta_min=1e-6)

    for epoch in range(1, stage2_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc, tr_rec = train_one_epoch(model, train_loader, criterion, optimizer_s2, device)
        scheduler_s2.step()
        val_loss, val_acc, val_rec, val_spec, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_tb_recall"].append(tr_rec)
        history["val_tb_recall"].append(val_rec)
        history["val_specificity"].append(val_spec)
        history["val_tb_f1"].append(val_f1)

        print(
            f"Epoch [{global_epoch:02d}/{total_epochs:02d}] (S2) - "
            f"Loss: {tr_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {tr_acc*100:.1f}%/{val_acc*100:.1f}% | "
            f"TB Recall: {tr_rec*100:.1f}%/{val_rec*100:.1f}% | "
            f"F1: {val_f1:.4f} | {elapsed:.1f}s"
        )

        # Early Stopping & Checkpoint check
        if (val_rec > best_val_tb_recall) or (val_rec == best_val_tb_recall and val_loss < best_val_loss):
            best_val_tb_recall = val_rec
            best_val_loss = val_loss
            best_epoch = global_epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_tb_recall": best_val_tb_recall,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_specificity": val_spec,
                    "class_names": CLASS_NAMES,
                    "architecture": "efficientnet_b0",
                    "class_weights": class_weights.tolist(),
                },
                best_model_path,
            )
            print(f"  --> Saved new best checkpoint: TB Recall={best_val_tb_recall*100:.1f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early stopping triggered at epoch {global_epoch} (patience={patience} reached).")
                break

    # Save training curves plot
    plot_training_curves(history, save_path=os.path.join(results_dir, "training_curves.png"))
    print(f"\n[✓] Training curves saved to: {os.path.join(results_dir, 'training_curves.png')}")

    # Save training history JSON
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_tb_recall": best_val_tb_recall,
                "best_val_loss": best_val_loss,
            },
            f,
            indent=2,
        )
    print(f"[✓] Training history saved to: {history_path}")
    print(f"[✓] Best model checkpoint: {best_model_path} (Epoch {best_epoch}, TB Recall: {best_val_tb_recall*100:.1f}%)")

    return {
        "best_epoch": best_epoch,
        "best_val_tb_recall": best_val_tb_recall,
        "best_model_path": best_model_path,
        "history": history,
    }


def main() -> None:
    """CLI Entrypoint for Model Training."""
    parser = argparse.ArgumentParser(description="Train TB-Screen AI Model")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Output directory for curves and metrics")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="Stage 1 head warmup epochs")
    parser.add_argument("--stage2_epochs", type=int, default=7, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr_stage1", type=float, default=1e-3, help="Stage 1 head LR")
    parser.add_argument("--lr_stage2", type=float, default=1e-4, help="Stage 2 fine-tuning LR")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    args = parser.parse_args()

    # Input validation
    if args.stage1_epochs < 0:
        parser.error("--stage1_epochs must be non-negative (>= 0).")
    if args.stage2_epochs < 1:
        parser.error("--stage2_epochs must be at least 1.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be a positive integer.")
    if args.lr_stage1 <= 0 or args.lr_stage2 <= 0:
        parser.error("Learning rates (--lr_stage1, --lr_stage2) must be positive floats.")
    if args.patience < 1:
        parser.error("--patience must be at least 1.")

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        seed=args.seed,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
