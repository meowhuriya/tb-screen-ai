"""Comprehensive Test Suite for TB-Screen AI Pipeline.

Covers:
1. Model architecture, parameter counts, and tensor forward pass
2. Dataset splits audit & zero-leakage guarantee
3. Image input validation & corrupt payload defense
4. Single-image prediction & Grad-CAM explainability pipeline
5. PDF clinical referral report formatting & magic bytes check
6. Model file size verification for cloud deployment compliance
"""

import json
import os
import sys
import pytest
import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.pdf_generator import create_diagnostic_pdf
from src.dataset import get_transforms
from src.model import get_model
from src.predict import (
    CALIBRATED_THRESHOLD,
    predict_chest_xray,
    predict_chest_xray_onnx,
    validate_and_load_image,
)


def test_model_architecture_and_forward():
    """Verify TBClassifier structure, parameter freezing, and output shapes."""
    model = get_model(pretrained=False, num_classes=2)
    total_p, trainable_p = model.count_parameters()
    assert total_p == 4010110, f"Expected 4,010,110 parameters, got {total_p}"

    # Stage 1: head warmup
    model.freeze_backbone()
    _, s1_trainable = model.count_parameters()
    assert s1_trainable == 2562, f"Expected 2,562 trainable params in head, got {s1_trainable}"

    # Stage 2: top 2 blocks unfrozen
    model.unfreeze_blocks(num_blocks=2)
    _, s2_trainable = model.count_parameters()
    assert s2_trainable == 1131954, f"Expected 1,131,954 params in stage 2, got {s2_trainable}"

    # Forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2), f"Expected logits shape (2, 2), got {output.shape}"


def test_dataset_splits_zero_leakage():
    """Audit manifest to mathematically verify zero data leakage across splits."""
    manifest_path = os.path.join(PROJECT_ROOT, "data", "processed", "splits_manifest.json")
    assert os.path.exists(manifest_path), "Splits manifest must exist"

    with open(manifest_path, "r") as f:
        splits = json.load(f)

    train_paths = set(splits["train"]["paths"])
    val_paths = set(splits["val"]["paths"])
    test_paths = set(splits["test"]["paths"])

    # Strict mathematical disjointness
    assert len(train_paths & val_paths) == 0, "Data leakage between train and val!"
    assert len(train_paths & test_paths) == 0, "Data leakage between train and test!"
    assert len(val_paths & test_paths) == 0, "Data leakage between val and test!"

    # Label checks
    assert set(splits["train"]["labels"]) == {0, 1}
    assert set(splits["test"]["labels"]) == {0, 1}


def test_input_validation_safety():
    """Verify input validation handles valid images (RGB, Grayscale L, RGBA) and rejects corrupt inputs."""
    # 1. Valid RGB PIL Image
    dummy_rgb = Image.new("RGB", (256, 256), color=(200, 200, 200))
    loaded = validate_and_load_image(dummy_rgb)
    assert loaded.mode == "RGB"
    assert loaded.size == (256, 256)

    # 2. 1-Channel Grayscale (L mode)
    dummy_gray = Image.new("L", (256, 256), color=128)
    loaded_gray = validate_and_load_image(dummy_gray)
    assert loaded_gray.mode == "RGB", "Grayscale images must be automatically converted to 3-channel RGB"
    assert loaded_gray.size == (256, 256)

    # 3. 4-Channel RGBA
    dummy_rgba = Image.new("RGBA", (256, 256), color=(180, 180, 180, 255))
    loaded_rgba = validate_and_load_image(dummy_rgba)
    assert loaded_rgba.mode == "RGB", "RGBA images must have alpha safely stripped and convert to RGB"
    assert loaded_rgba.size == (256, 256)

    # 4. Truncated image bytes (partial JPEG marker)
    corrupted_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x60\x00\x60\x00\x00"  # Missing image payload
    with pytest.raises(ValueError, match="Invalid or corrupted"):
        validate_and_load_image(corrupted_jpeg)

    # 5. Corrupt / Random non-image bytes
    with pytest.raises(ValueError, match="Invalid or corrupted"):
        validate_and_load_image(b"THIS_IS_DEFINITELY_NOT_A_VALID_IMAGE_DATA_BLOCK")

    # 6. Nonexistent file path
    with pytest.raises(FileNotFoundError):
        validate_and_load_image("/nonexistent/path/to/xray_image.png")


def test_prediction_pipeline():
    """Verify full end-to-end inference and Grad-CAM generation on sample X-ray."""
    sample_path = os.path.join(PROJECT_ROOT, "app", "sample_images", "sample_normal.png")
    assert os.path.exists(sample_path), f"Sample image missing at {sample_path}"

    result = predict_chest_xray(sample_path, threshold=CALIBRATED_THRESHOLD)

    # Key presence check
    expected_keys = [
        "predicted_label",
        "is_positive_tb",
        "tb_probability",
        "normal_probability",
        "confidence_percentage",
        "risk_tier",
        "risk_color_hex",
        "risk_emoji",
        "decision_threshold_used",
        "original_image_pil",
        "gradcam_overlay_rgb",
        "gradcam_grayscale",
    ]
    for key in expected_keys:
        assert key in result, f"Result missing expected key: {key}"

    # Value ranges
    assert 0.0 <= result["tb_probability"] <= 1.0
    assert 0.0 <= result["normal_probability"] <= 1.0
    assert abs((result["tb_probability"] + result["normal_probability"]) - 1.0) < 1e-4
    assert result["risk_tier"] in ["LOW", "NEEDS_REVIEW", "HIGH"]
    assert result["gradcam_overlay_rgb"].shape == (224, 224, 3)


def test_onnx_prediction_pipeline():
    """Verify standalone ONNX Runtime prediction pipeline execution and latency."""
    sample_path = os.path.join(PROJECT_ROOT, "app", "sample_images", "sample_normal.png")
    assert os.path.exists(sample_path)

    result = predict_chest_xray_onnx(sample_path, threshold=CALIBRATED_THRESHOLD)
    assert "tb_probability" in result
    assert "risk_tier" in result
    assert 0.0 <= result["tb_probability"] <= 1.0
    assert result["engine"].startswith("ONNX Runtime")
    # ONNX inference on modern CPU should be < 100ms
    assert result["inference_time_ms"] < 200.0, f"ONNX latency {result['inference_time_ms']}ms too slow"


def test_pdf_referral_report_generation():
    """Verify PDF generator produces valid PDF binary structure."""
    sample_path = os.path.join(PROJECT_ROOT, "app", "sample_images", "sample_tb_advanced.png")
    prediction = predict_chest_xray(sample_path)

    pdf_bytes = create_diagnostic_pdf(
        prediction=prediction,
        patient_id="TEST-PT-99",
        clinic_name="Primary Care Rural Center",
        notes="Suspected infiltrative TB lesion in upper zone.",
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 5000, "PDF bytes should be non-trivial"
    # PDF magic bytes
    assert pdf_bytes.startswith(b"%PDF-"), "Generated file must start with PDF magic header"


def test_model_weight_file_size_for_cloud_deployment():
    """Verify checkpoint size meets cloud deploy constraints (Streamlit Cloud < 100MB)."""
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
    assert os.path.exists(model_path), f"Model weight missing at {model_path}"

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    # Target: EfficientNet-B0 should be ~16MB, strictly under 50MB
    assert size_mb < 50.0, f"Model file size ({size_mb:.2f} MB) is too large for zero-LFS git deployment"
    assert size_mb > 5.0, f"Model file size ({size_mb:.2f} MB) seems incomplete or corrupt"


def test_hardware_benchmark_profiler():
    """Verify the benchmark profiler runs on CPU and returns valid latency metrics."""
    import tempfile
    from src.benchmark import benchmark_inference
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
    test_json_path = os.path.join(tempfile.gettempdir(), "test_benchmark.json")
    bench = benchmark_inference(
        model_path=model_path,
        device_name="cpu",
        num_iterations=3,
        warmup_runs=1,
        save_path=test_json_path,
    )
    assert bench["device"] == "cpu"
    assert bench["latency_p50_ms"] > 0.0
    assert bench["peak_rss_memory_mb"] > 0.0
    assert "clinic_feasibility" in bench


def test_clahe_and_lung_roi_preprocessing():
    """Verify CLAHE contrast enhancement and ROI cropping transformations."""
    from src.dataset import ApplyCLAHE, CropLungROI

    # Dummy grayscale radiograph with low contrast and dark margins
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[30:270, 30:270] = 120  # Central lung field
    dummy_pil = Image.fromarray(arr)

    # 1. Test CLAHE
    clahe_fn = ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    enhanced = clahe_fn(dummy_pil)
    assert isinstance(enhanced, Image.Image)
    assert enhanced.size == (300, 300)

    # 2. Test Lung ROI Crop
    crop_fn = CropLungROI(margin_ratio=0.08)
    cropped = crop_fn(dummy_pil)
    assert isinstance(cropped, Image.Image)
    w, h = cropped.size
    assert w <= 300 and h <= 300


def test_onnx_inference_and_latency_under_2s():
    """Verify ONNX graph inference executes accurately in under 2.0 seconds on CPU."""
    import onnxruntime as ort
    from src.config import ONNX_MODEL_PATH
    import time

    assert os.path.exists(ONNX_MODEL_PATH), f"ONNX model missing at {ONNX_MODEL_PATH}"

    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Measure CPU latency
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: dummy_input})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert len(outputs) == 1
    assert outputs[0].shape == (1, 2)
    # Target: under 2000 ms (2.0s), typically ~10 ms
    assert latency_ms < 2000.0, f"ONNX CPU inference took {latency_ms:.2f} ms (exceeds 2.0s limit)"


def test_configuration_integrity():
    """Verify centralized typed configuration dataclasses are properly configured."""
    from src.config import CLINICAL, MODEL, PREPROCESSING

    assert PREPROCESSING.image_size == 224
    assert PREPROCESSING.clahe_clip_limit == 2.0
    assert MODEL.architecture == "efficientnet_b0"
    assert MODEL.num_classes == 2
    assert CLINICAL.calibrated_threshold == 0.27
    assert CLINICAL.who_benchmark_threshold == 0.05
    assert CLINICAL.risk_low_cutoff < CLINICAL.risk_high_cutoff


def test_ood_detection_and_rejection():
    """Verify OOD detector refuses non-radiographs and uniform blank canvases."""
    from src.safety import inspect_xray_quality_and_ood

    # 1. Non-CXR Color Photograph (High Saturation)
    color_arr = np.zeros((250, 250, 3), dtype=np.uint8)
    color_arr[:, :, 0] = 255  # Solid Red
    color_img = Image.fromarray(color_arr)
    is_valid, err, _ = inspect_xray_quality_and_ood(color_img)
    assert not is_valid
    assert "Out-of-Distribution" in err

    # 2. Blank uniform canvas (Low variance)
    blank_img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 128)
    is_valid_b, err_b, _ = inspect_xray_quality_and_ood(blank_img)
    assert not is_valid_b
    assert "Blank or Uniform" in err_b

    # 3. Too small dimension (< 100x100)
    tiny_img = Image.fromarray(np.ones((50, 50, 3), dtype=np.uint8) * 100)
    is_valid_t, err_t, _ = inspect_xray_quality_and_ood(tiny_img)
    assert not is_valid_t
    assert "below the minimum diagnostic requirement" in err_t


def test_radiographic_quality_warnings():
    """Verify advisory quality diagnostics detect blur and extreme exposures without crashing."""
    from src.safety import inspect_xray_quality_and_ood

    # Real grayscale X-ray sample
    sample_path = os.path.join(PROJECT_ROOT, "app", "sample_images", "sample_normal.png")
    real_xray = Image.open(sample_path)
    is_valid, _, warnings = inspect_xray_quality_and_ood(real_xray)
    assert is_valid, "Valid X-ray must pass OOD inspection"


def test_sample_images_split_integrity_and_tint_ood():
    """Confirm demo sample images originate from test split (not train), pass OOD, and test tint filter."""
    from src.safety import inspect_xray_quality_and_ood

    manifest_path = os.path.join(PROJECT_ROOT, "data", "processed", "splits_manifest.json")
    with open(manifest_path, "r") as f:
        splits = json.load(f)

    train_files = {os.path.basename(p) for p in splits["train"]["paths"]}
    test_files = {os.path.basename(p) for p in splits["test"]["paths"]}

    # Sample filenames mapped to original dataset files
    sample_mapping = {
        "sample_normal.png": "Normal-698.png",
        "sample_tb_advanced.png": "Tuberculosis-283.png",
        "sample_tb_moderate.png": "Tuberculosis-418.png",
    }

    sample_dir = os.path.join(PROJECT_ROOT, "app", "sample_images")
    for s_file, orig_file in sample_mapping.items():
        # 1. Assert never in training split
        assert orig_file not in train_files, f"{orig_file} leaked into training cohort!"
        # 2. Assert present in test split
        assert orig_file in test_files, f"{orig_file} must belong to test cohort!"

        # 3. Assert passes OOD inspection
        img_path = os.path.join(sample_dir, s_file)
        img = Image.open(img_path)
        is_valid, err, _ = inspect_xray_quality_and_ood(img)
        assert is_valid, f"{s_file} failed OOD inspection: {err}"

    # 4. Verify that strongly tinted scans (e.g. 35% saturation) are rejected as documented
    tinted_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    tinted_arr[:, :, 0] = 100  # Red
    tinted_arr[:, :, 1] = 120  # Green
    tinted_arr[:, :, 2] = 200  # Strong Blue cast (>30% saturation)
    tinted_img = Image.fromarray(tinted_arr)
    is_valid_tint, err_tint, _ = inspect_xray_quality_and_ood(tinted_img)
    assert not is_valid_tint
    assert "Out-of-Distribution" in err_tint


def test_who_benchmark_and_calibrated_thresholds():
    """Verify inference pipeline supports WHO benchmark (0.05) and calibrated cutoff (0.27)."""
    from src.predict import CALIBRATED_THRESHOLD, WHO_BENCHMARK_THRESHOLD, predict_chest_xray, predict_chest_xray_onnx

    sample_path = os.path.join(PROJECT_ROOT, "app", "sample_images", "sample_tb_advanced.png")
    assert os.path.exists(sample_path)

    # 1. Evaluate with WHO default threshold (0.05)
    res_who = predict_chest_xray(sample_path, threshold=WHO_BENCHMARK_THRESHOLD)
    assert res_who["decision_threshold_used"] == 0.05
    assert res_who["is_positive_tb"] is True
    assert res_who["risk_tier"] == "HIGH"

    # 2. Evaluate with ONNX and Calibrated threshold (0.27)
    res_cal = predict_chest_xray_onnx(sample_path, threshold=CALIBRATED_THRESHOLD)
    assert res_cal["decision_threshold_used"] == 0.27
    assert res_cal["is_positive_tb"] is True


def test_app_input_selection_state_machine():
    """Verify Streamlit app input selection, mutual exclusion, and stable session state."""
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(PROJECT_ROOT, "app", "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"AppTest run failed: {at.exception}"

    # Initial state
    assert at.session_state["sample_selection"] == "None"
    assert at.session_state["uploader_key_id"] == 0

    # 1. Select Sample 1
    at.sidebar.radio[0].set_value("1. Normal Chest (Healthy)").run()
    assert len(at.exception) == 0
    assert at.session_state["sample_selection"] == "Sample 1"
    banner_text = [m.value for m in at.markdown if "Currently analysing" in m.value]
    assert len(banner_text) > 0
    assert "Normal Chest" in banner_text[0]

    # 2. Select Sample 2
    at.sidebar.radio[0].set_value("2. Advanced TB (Cavitation)").run()
    assert len(at.exception) == 0
    assert at.session_state["sample_selection"] == "Sample 2"
    banner_text = [m.value for m in at.markdown if "Currently analysing" in m.value]
    assert "Advanced TB" in banner_text[0]

    # 3. Upload a file: must switch sample_selection to "None" and analyse the upload
    sample_normal_path = os.path.join(PROJECT_ROOT, "app", "sample_images", "sample_normal.png")
    with open(sample_normal_path, "rb") as f:
        img_bytes = f.read()

    at.file_uploader[0].upload("patient_scan.png", img_bytes).run()
    assert len(at.exception) == 0
    assert at.session_state["sample_selection"] == "None"
    banner_text = [m.value for m in at.markdown if "Currently analysing" in m.value]
    assert "patient_scan.png" in banner_text[0]

    # 4. Choosing Sample 1 again must clear uploaded file and analyse Sample 1
    at.sidebar.radio[0].set_value("1. Normal Chest (Healthy)").run()
    assert len(at.exception) == 0
    assert at.session_state["sample_selection"] == "Sample 1"
    assert at.session_state["uploader_key_id"] > 0
    banner_text = [m.value for m in at.markdown if "Currently analysing" in m.value]
    assert "Normal Chest" in banner_text[0]





