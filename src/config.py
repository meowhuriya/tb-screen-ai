"""Centralized Configuration & Hyperparameter Definitions for TB-Screen AI.

Track: HI-01 - Medical Image Diagnostic Assistant
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# Project Directory Structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DATA_EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
SPLITS_MANIFEST_PATH = os.path.join(DATA_PROCESSED_DIR, "splits_manifest.json")
MONTGOMERY_MANIFEST_PATH = os.path.join(DATA_EXTERNAL_DIR, "montgomery_manifest.json")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "model.onnx")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "metrics.json")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Chest radiograph normalization and enhancement parameters."""
    image_size: int = 224
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    lung_roi_margin: float = 0.08
    min_image_dimension: int = 100
    max_image_filesize_mb: float = 10.0


@dataclass(frozen=True)
class ModelConfig:
    """Deep learning architecture & training hyperparameters."""
    architecture: str = "efficientnet_b0"
    num_classes: int = 2
    dropout_rate: float = 0.25
    hidden_dim: int = 256
    batch_size: int = 32
    learning_rate_head: float = 1e-3
    learning_rate_backbone: float = 1e-4
    stage1_epochs: int = 2
    stage2_epochs: int = 8
    focal_loss_gamma: float = 2.0
    seed: int = 42


@dataclass(frozen=True)
class ClinicalTriageConfig:
    """Public health screening cutoffs & traffic-light triage."""
    # Optimal calibrated operating threshold from Phase 3 evaluation
    calibrated_threshold: float = 0.27
    # WHO benchmark threshold ensuring Sensitivity >= 95%
    who_benchmark_threshold: float = 0.05
    # Standard default cutoff
    default_threshold: float = 0.50

    # Risk classification tiers
    risk_low_cutoff: float = 0.15     # < 15% -> Low Risk (Green)
    risk_high_cutoff: float = 0.60    # > 60% -> High Risk (Red)
    # Between 15% and 60% -> Needs Review (Yellow/Amber)

    supported_languages: List[str] = field(default_factory=lambda: ["en", "hi", "te"])


# Global singleton instances
PREPROCESSING = PreprocessingConfig()
MODEL = ModelConfig()
CLINICAL = ClinicalTriageConfig()
