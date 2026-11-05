"""Safety, Out-of-Distribution (OOD) Defense & Image Quality Diagnostics.

Track: HI-01 - Medical Image Diagnostic Assistant
Phase 5 Real-Clinic Safety Layer:
- Rejects non-chest radiographs (photographs, graphics, natural scenes, animals)
- Validates image dimensions (minimum 100x100 px, maximum 10 MB)
- Detects blank or low-contrast canvases (zero/minimal variance)
- Provides non-fatal clinical warnings for blur, underexposure, and overexposure
"""

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from src.config import PREPROCESSING


def validate_image_file_size(file_bytes: bytes, max_mb: float = PREPROCESSING.max_image_filesize_mb) -> Tuple[bool, Optional[str]]:
    """Verify raw file payload size is within clinical limits."""
    size_mb = len(file_bytes) / (1024.0 * 1024.0)
    if size_mb > max_mb:
        return False, f"File size ({size_mb:.1f} MB) exceeds maximum clinic threshold ({max_mb:.0f} MB)."
    if size_mb < 0.005:  # Under 5KB is corrupt or empty
        return False, "File size is too small (<5 KB) to contain a diagnostic radiograph."
    return True, None


def inspect_xray_quality_and_ood(
    image: Image.Image,
) -> Tuple[bool, Optional[str], List[str]]:
    """Perform deterministic OOD filtering and radiographic image quality checks.

    Args:
        image: PIL Image object.

    Returns:
        Tuple of:
          - is_valid_cxr: True if input is a plausible frontal chest radiograph.
          - ood_error_message: Failure explanation if OOD, else None.
          - quality_warnings: List of advisory clinical warnings (blur, exposure, low res).
    """
    quality_warnings: List[str] = []

    # 1. Dimension Check
    width, height = image.size
    min_dim = PREPROCESSING.min_image_dimension
    if width < min_dim or height < min_dim:
        return (
            False,
            f"Image dimensions ({width}x{height} px) are below the minimum diagnostic requirement "
            f"({min_dim}x{min_dim} px). Please provide a standard resolution chest radiograph.",
            quality_warnings,
        )

    # Convert to RGB numpy array
    rgb_arr = np.array(image.convert("RGB"))

    # 2. Color Saturation Check (Out-of-Distribution Natural Photo Defense)
    # Chest radiographs are transmission scans with near-zero chroma/saturation.
    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    sat_channel = hsv[:, :, 1].astype(float) / 255.0
    mean_saturation = float(np.mean(sat_channel))
    high_sat_pixel_ratio = float(np.mean(sat_channel > 0.35))

    # Reject if substantial color saturation is detected
    if mean_saturation > 0.20 or high_sat_pixel_ratio > 0.15:
        return (
            False,
            "⚠️ Out-of-Distribution Input Detected: The uploaded image contains vivid color saturation "
            f"(mean saturation: {mean_saturation*100:.1f}%), indicating a color photograph or graphic. "
            "TB-Screen AI strictly evaluates grayscale frontal chest radiographs (CXR).",
            quality_warnings,
        )

    # 3. Variance & Contrast Check (Blank Canvas / Noise Defense)
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    std_intensity = float(np.std(gray))
    if std_intensity < 8.0:
        return (
            False,
            "⚠️ Blank or Uniform Image: The uploaded file lacks radiographic contrast "
            f"(intensity standard deviation: {std_intensity:.1f} < 8.0). Please upload a valid radiograph.",
            quality_warnings,
        )

    # 4. Blur / Sharpness Check (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < 30.0:
        quality_warnings.append(
            f"⚠️ Blur Warning (Sharpness Score: {laplacian_var:.1f}): The radiograph is soft or out of focus. "
            "Fine apical reticular patterns and micro-nodules may be partially obscured."
        )

    # 5. Exposure / Penetration Warnings
    mean_intensity = float(np.mean(gray))
    if mean_intensity < 45.0:
        quality_warnings.append(
            f"⚠️ Underexposure Warning (Mean Density: {mean_intensity:.1f}/255): Radiograph is darkly penetrated. "
            "Lower lung bases and retrocardiac spaces may have reduced diagnostic visibility."
        )
    elif mean_intensity > 210.0:
        quality_warnings.append(
            f"⚠️ Overexposure Warning (Mean Density: {mean_intensity:.1f}/255): Radiograph appears washed out / over-penetrated. "
            "Subtle ground-glass parenchymal opacities may be burned out."
        )

    # 6. Low Resolution Advisory
    if width < 250 or height < 250:
        quality_warnings.append(
            f"ℹ️ Low Resolution Advisory ({width}x{height} px): Image is below recommended 512x512 digital CXR standard. "
            "Image was auto-upscaled for AI screening."
        )

    return True, None, quality_warnings
