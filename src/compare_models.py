"""Phase 2: Comprehensive Architecture Comparison, 5-Fold Stratified Cross-Validation & Ensembling.

Executes:
1. Data Preprocessing: Lung ROI cropping + CLAHE enhancement
2. 5-Fold Stratified Cross-Validation on the training corpus (Mean +- Std Dev)
3. Three-way Transfer Learning Comparison:
   - EfficientNet-B0 (Compound scaling, lightweight edge model)
   - DenseNet-121 (Dense connectivity, classic medical imaging baseline)
   - ResNet-50 (Deep residual network baseline)
4. Model Ensembling (Soft-voting of top 2 architectures)
5. Test-Time Augmentation (TTA) (Multi-scale averaging)
6. Export of results/model_comparison.csv and results/model_comparison.png
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
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import (
    ApplyCLAHE,
    ChestXRayDataset,
    CropLungROI,
    FocalLoss,
    IMAGENET_MEAN,
    IMAGENET_STD,
    compute_class_weights,
    get_device,
    get_transforms,
)
from src.model import TBClassifier, get_model


def train_model_quick(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    s1_epochs: int = 2,
    s2_epochs: int = 3,
    lr_s1: float = 1e-3,
    lr_s2: float = 1e-4,
) -> nn.Module:
    """Train a model through 2-stage transfer learning."""
    # Stage 1: Warmup
    model.freeze_backbone()
    opt_s1 = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_s1, weight_decay=1e-4)
    sched_s1 = CosineAnnealingLR(opt_s1, T_max=s1_epochs, eta_min=1e-5)

    for _ in range(s1_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_s1.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt_s1.step()
        sched_s1.step()

    # Stage 2: Fine-tuning
    model.unfreeze_blocks(num_blocks=2)
    opt_s2 = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_s2, weight_decay=1e-4)
    sched_s2 = CosineAnnealingLR(opt_s2, T_max=s2_epochs, eta_min=1e-6)

    best_state = None
    best_val_loss = float("inf")

    for _ in range(s2_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_s2.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt_s2.step()
        sched_s2.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Compute softmax probabilities for class 1 (TB)."""
    model.eval()
    all_probs = []
    softmax = nn.Softmax(dim=1)
    for x, _ in dataloader:
        x = x.to(device)
        p = softmax(model(x))[:, 1].cpu().numpy()
        all_probs.extend(p)
    return np.array(all_probs)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.27) -> Dict[str, float]:
    """Compute complete diagnostic metrics at a decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0.0))
    spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0.0))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0.0))
    auc_score = float(roc_auc_score(y_true, y_prob))

    return {
        "Accuracy": round(acc * 100, 2),
        "Sensitivity (Recall)": round(rec * 100, 2),
        "Specificity": round(spec * 100, 2),
        "Precision": round(prec * 100, 2),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc_score, 4),
    }


def run_5fold_cross_validation(
    train_paths: List[str],
    train_labels: List[int],
    device: torch.device,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, any]:
    """Execute 5-fold Stratified Cross-Validation on the training cohort."""
    print("\n" + "="*70)
    print("STEP 1: 5-FOLD STRATIFIED CROSS-VALIDATION (EfficientNet-B0 + CLAHE)")
    print("="*70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_transform, eval_transform = get_transforms(image_size=224, use_clahe=True, crop_roi=True)

    fold_metrics: Dict[str, List[float]] = {
        "Accuracy": [], "Sensitivity (Recall)": [], "Specificity": [],
        "Precision": [], "F1-Score": [], "ROC-AUC": [],
    }

    class_weights = compute_class_weights(train_labels).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    for fold, (t_idx, v_idx) in enumerate(skf.split(train_paths, train_labels), start=1):
        t0 = time.time()
        f_train_paths = [train_paths[i] for i in t_idx]
        f_train_labels = [train_labels[i] for i in t_idx]
        f_val_paths = [train_paths[i] for i in v_idx]
        f_val_labels = [train_labels[i] for i in v_idx]

        t_loader = DataLoader(
            ChestXRayDataset(f_train_paths, f_train_labels, transform=train_transform),
            batch_size=48, shuffle=True, num_workers=0,
        )
        v_loader = DataLoader(
            ChestXRayDataset(f_val_paths, f_val_labels, transform=eval_transform),
            batch_size=48, shuffle=False, num_workers=0,
        )

        model = get_model(architecture="efficientnet_b0", pretrained=True).to(device)
        model = train_model_quick(model, t_loader, v_loader, criterion, device, s1_epochs=2, s2_epochs=3)

        y_true = np.array(f_val_labels)
        y_prob = predict_probabilities(model, v_loader, device)
        m = evaluate_predictions(y_true, y_prob, threshold=0.27)

        for k, v in m.items():
            fold_metrics[k].append(v)

        elapsed = time.time() - t0
        print(f"Fold {fold}/{n_splits} - Acc: {m['Accuracy']}% | Sens: {m['Sensitivity (Recall)']}% | "
              f"Spec: {m['Specificity']}% | F1: {m['F1-Score']:.4f} | AUC: {m['ROC-AUC']:.4f} ({elapsed:.1f}s)")

    # Compute Mean +- Std
    cv_summary = {}
    print("\n5-FOLD CROSS-VALIDATION SUMMARY (MEAN +- STD):")
    for k, v_list in fold_metrics.items():
        mean_v = float(np.mean(v_list))
        std_v = float(np.std(v_list))
        cv_summary[k] = {"mean": round(mean_v, 4), "std": round(std_v, 4)}
        unit = "%" if "Score" not in k and "AUC" not in k else ""
        print(f"  • {k:<22}: {mean_v:.2f}{unit} ± {std_v:.2f}{unit}")

    return cv_summary


def test_time_augmentation_inference(
    model: nn.Module,
    image_paths: List[str],
    labels: List[int],
    device: torch.device,
) -> np.ndarray:
    """Predict probabilities with multi-view Test-Time Augmentation (TTA).
    
    Includes CropLungROI and ApplyCLAHE to match the training distribution.
    """
    # 3 clinical TTA views: standard, slight zoom (1.04), slight brightness boost
    view1_transform = transforms.Compose([
        CropLungROI(margin_ratio=0.08),
        ApplyCLAHE(clip_limit=2.0),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    view2_transform = transforms.Compose([
        CropLungROI(margin_ratio=0.08),
        ApplyCLAHE(clip_limit=2.0),
        transforms.Resize((236, 236)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    view3_transform = transforms.Compose([
        CropLungROI(margin_ratio=0.08),
        ApplyCLAHE(clip_limit=2.0),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    loader1 = DataLoader(ChestXRayDataset(image_paths, labels, transform=view1_transform), batch_size=48, shuffle=False)
    loader2 = DataLoader(ChestXRayDataset(image_paths, labels, transform=view2_transform), batch_size=48, shuffle=False)
    loader3 = DataLoader(ChestXRayDataset(image_paths, labels, transform=view3_transform), batch_size=48, shuffle=False)

    p1 = predict_probabilities(model, loader1, device)
    p2 = predict_probabilities(model, loader2, device)
    p3 = predict_probabilities(model, loader3, device)

    return (p1 + p2 + p3) / 3.0


def run_architecture_comparison(
    splits_manifest_path: str = "data/processed/splits_manifest.json",
    results_dir: str = "results",
    models_dir: str = "models",
) -> pd.DataFrame:
    """Train and benchmark EfficientNet-B0, DenseNet-121, ResNet-50, Ensembles, and TTA."""
    device = get_device()
    print(f"[*] Starting Multi-Model Comparison on: {device}")

    with open(splits_manifest_path, "r") as f:
        splits = json.load(f)

    train_paths = splits["train"]["paths"]
    train_labels = splits["train"]["labels"]
    val_paths = splits["val"]["paths"]
    val_labels = splits["val"]["labels"]

    # Load sanitized test set (excluding 3 leaked duplicates)
    raw_test_paths = splits["test"]["paths"]
    raw_test_labels = splits["test"]["labels"]

    # Filter out leaked indices
    leaked_filenames = {"Tuberculosis-509.png", "Tuberculosis-244.png", "Tuberculosis-467.png"}
    clean_test_paths = [p for p in raw_test_paths if os.path.basename(p) not in leaked_filenames]
    clean_test_labels = [l for p, l in zip(raw_test_paths, raw_test_labels) if os.path.basename(p) not in leaked_filenames]

    print(f"[*] Training: {len(train_paths)} | Val: {len(val_paths)} | Sanitized Test: {len(clean_test_paths)}")

    # 1. Run 5-Fold Stratified CV
    cv_summary = run_5fold_cross_validation(train_paths, train_labels, device=device)

    # 2. Train Architectures on Full Train Cohort
    train_transform, eval_transform = get_transforms(image_size=224, use_clahe=True, crop_roi=True)
    train_loader = DataLoader(ChestXRayDataset(train_paths, train_labels, transform=train_transform), batch_size=48, shuffle=True)
    val_loader = DataLoader(ChestXRayDataset(val_paths, val_labels, transform=eval_transform), batch_size=48, shuffle=False)
    test_loader = DataLoader(ChestXRayDataset(clean_test_paths, clean_test_labels, transform=eval_transform), batch_size=48, shuffle=False)

    class_weights = compute_class_weights(train_labels).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    architectures = ["efficientnet_b0", "densenet121", "resnet50"]
    trained_models: Dict[str, nn.Module] = {}
    model_probs: Dict[str, np.ndarray] = {}
    model_params_count: Dict[str, int] = {}

    print("\n" + "="*70)
    print("STEP 2: TRAINING & BENCHMARKING 3 CNN ARCHITECTURES (TRANSFER LEARNING)")
    print("="*70)

    for arch in architectures:
        t0 = time.time()
        print(f"\n[*] Training {arch.upper()} with CLAHE + Lung ROI Preprocessing...")
        model = get_model(architecture=arch, pretrained=True).to(device)
        total_p, _ = model.count_parameters()
        model_params_count[arch] = total_p

        model = train_model_quick(model, train_loader, val_loader, criterion, device, s1_epochs=2, s2_epochs=4)
        trained_models[arch] = model

        # Save checkpoint
        ckpt_path = os.path.join(models_dir, f"{arch}_checkpoint.pth")
        torch.save({"architecture": arch, "model_state_dict": model.state_dict()}, ckpt_path)

        probs = predict_probabilities(model, test_loader, device)
        model_probs[arch] = probs
        elapsed = time.time() - t0
        print(f"[✓] {arch.upper()} complete ({elapsed:.1f}s) | Params: {total_p:,}")

    y_true = np.array(clean_test_labels)

    # 3. Test-Time Augmentation (TTA) on top model (EfficientNet-B0)
    print("\n[*] Evaluating Test-Time Augmentation (TTA) on EfficientNet-B0...")
    tta_effnet_probs = test_time_augmentation_inference(
        trained_models["efficientnet_b0"], clean_test_paths, clean_test_labels, device
    )

    # 4. Ensemble (Top 2: EfficientNet-B0 + DenseNet-121)
    print("[*] Generating Soft-Voting Ensemble (EfficientNet-B0 + DenseNet-121)...")
    ensemble_probs = (model_probs["efficientnet_b0"] + model_probs["densenet121"]) / 2.0

    # 5. Ensemble + TTA
    print("[*] Evaluating Ensemble + TTA...")
    tta_dense_probs = test_time_augmentation_inference(
        trained_models["densenet121"], clean_test_paths, clean_test_labels, device
    )
    ensemble_tta_probs = (tta_effnet_probs + tta_dense_probs) / 2.0

    # 6. Tabulate Complete Comparison Table
    records = []
    comparisons = [
        ("EfficientNet-B0 (Single)", model_probs["efficientnet_b0"], model_params_count["efficientnet_b0"]),
        ("EfficientNet-B0 + TTA", tta_effnet_probs, model_params_count["efficientnet_b0"]),
        ("DenseNet-121 (Single)", model_probs["densenet121"], model_params_count["densenet121"]),
        ("ResNet-50 (Single)", model_probs["resnet50"], model_params_count["resnet50"]),
        ("Ensemble (EffNet + DenseNet)", ensemble_probs, model_params_count["efficientnet_b0"] + model_params_count["densenet121"]),
        ("Ensemble (EffNet + DenseNet) + TTA", ensemble_tta_probs, model_params_count["efficientnet_b0"] + model_params_count["densenet121"]),
    ]

    for name, p, param_c in comparisons:
        m = evaluate_predictions(y_true, p, threshold=0.27)
        rec = {
            "Model Pipeline": name,
            "Parameters": f"{param_c / 1e6:.2f}M",
            "Accuracy (%)": m["Accuracy"],
            "Sensitivity (%)": m["Sensitivity (Recall)"],
            "Specificity (%)": m["Specificity"],
            "Precision (%)": m["Precision"],
            "F1-Score": m["F1-Score"],
            "ROC-AUC": m["ROC-AUC"],
        }
        records.append(rec)

    df_results = pd.DataFrame(records)

    # Save CSV
    csv_path = os.path.join(results_dir, "model_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[✓] Model comparison CSV saved to: {csv_path}")

    # Save Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Metric Bars: Sensitivity vs Specificity
    models_short = [
        "EffNet-B0", "EffNet+TTA", "DenseNet121", "ResNet50", "Ensemble", "Ensemble+TTA"
    ]
    x_pos = np.arange(len(models_short))
    width = 0.35

    axes[0].bar(x_pos - width/2, df_results["Sensitivity (%)"], width, label="TB Sensitivity (%)", color="#d62728", alpha=0.85)
    axes[0].bar(x_pos + width/2, df_results["Specificity (%)"], width, label="Specificity (%)", color="#1f77b4", alpha=0.85)
    axes[0].axhline(y=95.0, color="gray", linestyle=":", label="WHO Target (95%)")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models_short, rotation=25, ha="right", fontsize=10)
    axes[0].set_ylim([90, 101])
    axes[0].set_title("Clinical Sensitivity vs Specificity by Architecture", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Metric Bars: ROC-AUC vs F1
    axes[1].bar(x_pos - width/2, df_results["ROC-AUC"], width, label="ROC-AUC", color="#2ca02c", alpha=0.85)
    axes[1].bar(x_pos + width/2, df_results["F1-Score"], width, label="F1-Score", color="#9467bd", alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models_short, rotation=25, ha="right", fontsize=10)
    axes[1].set_ylim([0.92, 1.005])
    axes[1].set_title("Discriminative Power (ROC-AUC & F1-Score)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("TB-Screen AI: Architecture Benchmark, Ensembling & TTA Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    chart_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✓] Model comparison chart saved to: {chart_path}")

    # Save CV results
    cv_json_path = os.path.join(results_dir, "cv_5fold_results.json")
    with open(cv_json_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"[✓] 5-Fold cross-validation results saved to: {cv_json_path}")

    print("\n" + "="*80)
    print("                 MODEL COMPARISON BENCHMARK TABLE")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)

    return df_results


def main() -> None:
    run_architecture_comparison()


if __name__ == "__main__":
    main()
