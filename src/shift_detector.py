"""Scanner-Shift Detection Module using Penultimate-Layer Mahalanobis Distance.

Addresses scanner-shift and domain-drift across clinical imaging centers:
1. Extracts 1280-dimensional penultimate embeddings from EfficientNet-B0.
2. Fits an empirical shrinkage covariance estimator (Ledoit-Wolf) on training embeddings.
3. Sets the calibration boundary tau at the 95th percentile of the untouched validation set.
4. Detects out-of-distribution scanner acquisitions on external cohorts (e.g. Montgomery County).
5. Provides lightweight serialization (<7MB) for sub-millisecond production inference.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.dataset import ChestXRayDataset, get_device, get_transforms
from src.model import TBClassifier


class MahalanobisShiftDetector:
    """Penultimate-embedding Mahalanobis distance scanner-shift detector."""

    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        precision: Optional[np.ndarray] = None,
        threshold: float = 0.0,
        percentile: float = 95.0,
    ) -> None:
        """Initialize detector.

        Args:
            mean: Feature mean vector of shape (D,).
            precision: Precision matrix (inverse covariance) of shape (D, D).
            threshold: Distance threshold tau determined from validation split.
            percentile: Percentile of validation distances used to select threshold.
        """
        self.mean_ = mean
        self.precision_ = precision
        self.threshold = float(threshold)
        self.percentile = float(percentile)

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.precision_ is not None

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        percentile: float = 95.0,
    ) -> "MahalanobisShiftDetector":
        """Fit Ledoit-Wolf covariance on training embeddings and select threshold on validation set.

        Args:
            X_train: Training embeddings of shape (N_train, D).
            X_val: Validation embeddings of shape (N_val, D).
            percentile: Percentile cutoff on validation distances (default: 95.0).

        Returns:
            Self.
        """
        print(f"Fitting Ledoit-Wolf shrinkage covariance on {X_train.shape[0]} training embeddings...")
        lw = LedoitWolf(assume_centered=False)
        lw.fit(X_train)

        self.mean_ = np.array(lw.location_, dtype=np.float32)
        self.precision_ = np.array(lw.precision_, dtype=np.float32)
        self.percentile = float(percentile)

        # Compute validation distances to set threshold without touching test data
        val_dists = self.compute_distance(X_val)
        self.threshold = float(np.percentile(val_dists, percentile))
        print(f"Shift detector fitted: tau ({percentile}th percentile val distance) = {self.threshold:.2f}")
        return self

    def compute_distance(self, X: np.ndarray) -> np.ndarray:
        """Compute squared Mahalanobis distance: (x - mu)^T Precision (x - mu).

        Args:
            X: Array of shape (N, D) or (D,).

        Returns:
            1D array of squared distances.
        """
        if not self.is_fitted:
            raise RuntimeError("Shift detector is not fitted yet.")

        X = np.atleast_2d(X)
        diff = X - self.mean_  # (N, D)
        # Vectorized quadratic form: sum_j diff_ij * (diff @ precision)_ij
        dist_sq = np.sum((diff @ self.precision_) * diff, axis=1)
        return np.maximum(dist_sq, 0.0)

    def predict_single(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Classify a single image embedding as familiar vs unfamiliar.

        Args:
            embedding: 1D array of shape (D,) or (1, D).

        Returns:
            Dictionary with distance, threshold, boolean flag, and clinical status text.
        """
        emb = np.asarray(embedding).reshape(1, -1)
        dist = float(self.compute_distance(emb)[0])
        is_unfamiliar = dist > self.threshold

        if is_unfamiliar:
            status_label = "Unfamiliar image source - results may be unreliable, run local calibration"
            badge_text = "⚠️ Unfamiliar Source"
            badge_color = "#ffc107"
        else:
            status_label = "Familiar image source"
            badge_text = "✅ Familiar Source"
            badge_color = "#28a745"

        return {
            "mahalanobis_distance": round(dist, 2),
            "threshold": round(self.threshold, 2),
            "is_unfamiliar": is_unfamiliar,
            "status_label": status_label,
            "badge_text": badge_text,
            "badge_color": badge_color,
        }

    def save(self, filepath: str) -> None:
        """Serialize detector parameters to a compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            mean=self.mean_,
            precision=self.precision_,
            threshold=np.array([self.threshold], dtype=np.float32),
            percentile=np.array([self.percentile], dtype=np.float32),
        )
        print(f"Saved shift detector to {filepath} ({os.path.getsize(filepath) / (1024*1024):.2f} MB)")

    @classmethod
    def load(cls, filepath: str) -> "MahalanobisShiftDetector":
        """Load detector from .npz file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Shift detector file not found: {filepath}")
        data = np.load(filepath)
        return cls(
            mean=data["mean"],
            precision=data["precision"],
            threshold=float(data["threshold"][0]),
            percentile=float(data["percentile"][0]),
        )


_CACHED_DETECTOR: Optional[MahalanobisShiftDetector] = None


def get_cached_shift_detector(
    detector_path: str = "models/shift_detector.npz",
) -> Optional[MahalanobisShiftDetector]:
    """Retrieve singleton cached shift detector if available."""
    global _CACHED_DETECTOR
    if _CACHED_DETECTOR is not None:
        return _CACHED_DETECTOR

    full_path = os.path.join(PROJECT_ROOT, detector_path) if not os.path.isabs(detector_path) else detector_path
    if os.path.exists(full_path):
        try:
            _CACHED_DETECTOR = MahalanobisShiftDetector.load(full_path)
            return _CACHED_DETECTOR
        except Exception as e:
            print(f"Warning: Failed to load shift detector from {full_path}: {e}")
            return None
    return None


def extract_embeddings_from_dataset(
    model: TBClassifier,
    paths: List[str],
    labels: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract penultimate 1280-dimensional feature representations for a list of image paths."""
    _, eval_transform = get_transforms(image_size=224, use_clahe=True, crop_roi=True)
    dataset = ChestXRayDataset(image_paths=paths, labels=labels, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            feats = model.extract_features(images).cpu().numpy()
            embeddings.append(feats)

    return np.vstack(embeddings).astype(np.float32)


def run_shift_detection_experiment(
    splits_manifest_path: str = "data/processed/splits_manifest.json",
    montgomery_manifest_path: str = "data/external/montgomery_manifest.json",
    model_path: str = "models/best_model.pth",
    output_detector_path: str = "models/shift_detector.npz",
    output_results_path: str = "results/shift_detection.json",
    output_plot_path: str = "results/shift_detection.png",
) -> Dict[str, Any]:
    """Execute complete Phase 1 Scanner-Shift detection training, evaluation, and documentation."""
    device = get_device()
    print(f"Initializing Scanner-Shift Detector on device: {device}")

    # 1. Load production model
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = TBClassifier(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Load manifests
    if not os.path.isabs(splits_manifest_path):
        splits_manifest_path = os.path.join(PROJECT_ROOT, splits_manifest_path)
    with open(splits_manifest_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    embeddings_cache_path = os.path.join(PROJECT_ROOT, "data", "processed", "embeddings_cache.npz")
    if os.path.exists(embeddings_cache_path):
        print(f"Loading cached feature embeddings from {embeddings_cache_path}...")
        cached_data = np.load(embeddings_cache_path)
        X_train = cached_data["X_train"]
        X_val = cached_data["X_val"]
        X_test = cached_data["X_test"]
        X_mont = cached_data["X_mont"]
        print(f"Loaded: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}, mont {X_mont.shape}")
    else:
        # 3. Extract train, val, and test embeddings
        print("Extracting training set embeddings (2,940 images)...")
        t0 = time.perf_counter()
        X_train = extract_embeddings_from_dataset(
            model, splits["train"]["paths"], splits["train"]["labels"], device=device
        )
        t_train = time.perf_counter() - t0
        print(f"Training embeddings extracted: shape {X_train.shape} in {t_train:.1f}s")

        print("Extracting validation set embeddings (630 images)...")
        X_val = extract_embeddings_from_dataset(
            model, splits["val"]["paths"], splits["val"]["labels"], device=device
        )

        print("Extracting untouched in-domain test set embeddings (630 images)...")
        X_test = extract_embeddings_from_dataset(
            model, splits["test"]["paths"], splits["test"]["labels"], device=device
        )

        # 4. Extract Montgomery external cohort
        if not os.path.isabs(montgomery_manifest_path):
            montgomery_manifest_path = os.path.join(PROJECT_ROOT, montgomery_manifest_path)
        with open(montgomery_manifest_path, "r", encoding="utf-8") as f:
            mont_data = json.load(f)
        print(f"Extracting Montgomery external cohort embeddings ({len(mont_data['paths'])} images)...")
        X_mont = extract_embeddings_from_dataset(
            model, mont_data["paths"], [0] * len(mont_data["paths"]), device=device
        )

        os.makedirs(os.path.dirname(embeddings_cache_path), exist_ok=True)
        np.savez_compressed(
            embeddings_cache_path,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            X_mont=X_mont,
        )
        print(f"Cached feature embeddings saved to {embeddings_cache_path}")

    # 5. Fit Mahalanobis detector on train only; threshold from val only
    detector = MahalanobisShiftDetector()
    detector.fit(X_train=X_train, X_val=X_val, percentile=95.0)

    # 6. Evaluate distances on test and external cohorts
    d_train = detector.compute_distance(X_train)
    d_val = detector.compute_distance(X_val)
    d_test = detector.compute_distance(X_test)
    d_mont = detector.compute_distance(X_mont)

    # Metric computations
    val_flagged_ratio = float(np.mean(d_val > detector.threshold))
    test_flagged_ratio = float(np.mean(d_test > detector.threshold))
    mont_flagged_ratio = float(np.mean(d_mont > detector.threshold))

    # AUROC for separating in-domain test (0 = familiar) from Montgomery (1 = unfamiliar)
    y_true = np.concatenate([np.zeros(len(d_test)), np.ones(len(d_mont))])
    y_scores = np.concatenate([d_test, d_mont])
    shift_auroc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    results = {
        "method": "Mahalanobis Distance with Ledoit-Wolf Shrinkage Covariance",
        "embedding_layer": "EfficientNet-B0 penultimate features (1,280 dimensions)",
        "train_samples_fitted": int(X_train.shape[0]),
        "validation_samples_calibrated": int(X_val.shape[0]),
        "validation_percentile_cutoff": detector.percentile,
        "calibrated_shift_threshold_tau": round(detector.threshold, 2),
        "in_domain_validation_flagged_percent": round(val_flagged_ratio * 100.0, 2),
        "in_domain_test_flagged_percent": round(test_flagged_ratio * 100.0, 2),
        "montgomery_external_flagged_percent": round(mont_flagged_ratio * 100.0, 2),
        "shift_separation_auroc": round(shift_auroc, 4),
        "distance_statistics": {
            "train": {
                "mean": round(float(np.mean(d_train)), 2),
                "std": round(float(np.std(d_train)), 2),
                "median": round(float(np.median(d_train)), 2),
            },
            "val": {
                "mean": round(float(np.mean(d_val)), 2),
                "std": round(float(np.std(d_val)), 2),
                "median": round(float(np.median(d_val)), 2),
            },
            "test": {
                "mean": round(float(np.mean(d_test)), 2),
                "std": round(float(np.std(d_test)), 2),
                "median": round(float(np.median(d_test)), 2),
            },
            "montgomery": {
                "mean": round(float(np.mean(d_mont)), 2),
                "std": round(float(np.std(d_mont)), 2),
                "median": round(float(np.median(d_mont)), 2),
            },
        },
        "clinical_safeguard_summary": (
            f"The Mahalanobis scanner-shift detector flagged {test_flagged_ratio * 100.0:.2f}% of in-domain test scans "
            f"(target ~5%) vs {mont_flagged_ratio * 100.0:.2f}% of Montgomery scans, achieving an AUROC of {shift_auroc:.4f}. "
            f"This flags unfamiliar digitized radiographs before unreliable inference occurs."
        ),
    }

    # 7. Save model and results
    if not os.path.isabs(output_detector_path):
        output_detector_path = os.path.join(PROJECT_ROOT, output_detector_path)
    detector.save(output_detector_path)

    if not os.path.isabs(output_results_path):
        output_results_path = os.path.join(PROJECT_ROOT, output_results_path)
    os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved shift detection metrics to {output_results_path}")

    # 8. Plot evaluation figure
    if not os.path.isabs(output_plot_path):
        output_plot_path = os.path.join(PROJECT_ROOT, output_plot_path)
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Distance Distribution Histograms
    ax0 = axes[0]
    bins = np.linspace(min(np.min(d_test), np.min(d_mont)), max(np.percentile(d_mont, 98), detector.threshold * 1.5), 40)
    ax0.hist(d_test, bins=bins, alpha=0.6, color="#1f77b4", label=f"In-Domain Test (N={len(d_test)})", density=True)
    ax0.hist(d_mont, bins=bins, alpha=0.6, color="#d62728", label=f"Montgomery Cohort (N={len(d_mont)})", density=True)
    ax0.axvline(detector.threshold, color="#2ca02c", linestyle="--", linewidth=2.2, label=f"95% Val Cutoff (τ = {detector.threshold:.0f})")
    ax0.set_xlabel("Penultimate Mahalanobis Distance (d_M^2)", fontsize=11)
    ax0.set_ylabel("Probability Density", fontsize=11)
    ax0.set_title("Feature Distance Distribution under Domain Shift", fontsize=12, fontweight="bold")
    ax0.legend(loc="upper right", frameon=True)
    ax0.grid(True, linestyle=":", alpha=0.6)

    # Right: Shift Detection ROC Curve
    ax1 = axes[1]
    ax1.plot(fpr, tpr, color="#ff7f0e", linewidth=2.5, label=f"Shift ROC-AUC = {shift_auroc:.4f}")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Random Guess (0.50)")
    ax1.scatter([test_flagged_ratio], [mont_flagged_ratio], color="#2ca02c", s=70, zorder=5, label=f"Operating Point (FPR={test_flagged_ratio:.1%}, TPR={mont_flagged_ratio:.1%})")
    ax1.set_xlabel("False Positive Rate (In-Domain Test Flagged)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Montgomery Shift Flagged)", fontsize=11)
    ax1.set_title("Scanner-Shift Discrimination ROC Curve", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()
    print(f"Saved shift detection figure to {output_plot_path}")

    return results


if __name__ == "__main__":
    res = run_shift_detection_experiment()
    print("\n--- SHIFT DETECTION RESULTS ---")
    print(json.dumps(res, indent=2))
