---
title: TB-Screen AI | Frontline Medical Diagnostic Assistant
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.66.1"
app_file: app/app.py
pinned: false
license: mit
---

# TB-Screen AI: Medical Image Diagnostic Assistant
### Problem Statement: HI-01 — Medical Image Diagnostic Assistant
**Hackathon:** Neural Storm | **Target Environment:** Resource-Constrained Rural Primary Health Centers (PHCs)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Community%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://tb-screen-ai-hrgejmak9ujcwfm5q4duap.streamlit.app)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Spaces%20Deployment-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%207.19ms-005CED?style=flat-square&logo=onnx&logoColor=white)](results/onnx_benchmark.json)
[![Test ROC-AUC](https://img.shields.io/badge/Test%20ROC--AUC-0.9780%20%2895%25%20CI%29-success?style=flat-square)](results/metrics.json)
[![WHO Sensitivity](https://img.shields.io/badge/WHO%20Sens%20Benchmark-95.10%25-brightgreen?style=flat-square)](results/metrics.json)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## Table of Contents
1. [Problem Statement: HI-01](#1-problem-statement-hi-01)
2. [Team Information](#2-team-information)
3. [Medical Advisor](#3-medical-advisor)
4. [Problem Description: Why TB Screening Fails in Rural Clinics](#4-problem-description-why-tb-screening-fails-in-rural-clinics)
5. [Our Approach & Architectural Pipeline](#5-our-approach--architectural-pipeline)
6. [Tech Stack & Libraries](#6-tech-stack--libraries)
7. [Setup and Installation](#7-setup-and-installation)
8. [How to Run Locally](#8-how-to-run-locally)
9. [Screenshots and Visual Explainability](#9-screenshots-and-visual-explainability)
10. [Model Performance & Rigorous Benchmarks](#10-model-performance--rigorous-benchmarks)
    - [In-Domain Test Set & 95% Bootstrap CIs](#101-in-domain-test-set-evaluation-with-95-bootstrap-cis)
    - [Sensitivity vs. Specificity Trade-Off Analysis](#102-decision-threshold-trade-off-analysis)
    - [Probability Calibration (Temperature Scaling)](#103-probability-calibration-temperature-scaling)
    - [Multi-Architecture & Ensemble Benchmark](#104-multi-architecture--ensemble-benchmark)
    - [Cross-Validation (5-Fold Stratified)](#105-5-fold-stratified-cross-validation)
    - [External Validation: Montgomery County NIH Dataset](#106-external-validation-montgomery-county-nih-dataset)
    - [Grad-CAM Error Analysis on Failure Cases](#107-grad-cam-diagnostic-error-analysis)
    - [Hardware Efficiency & Measured Batch Speed](#108-hardware-efficiency--measured-batch-screening-speed)
11. [Limitations and Clinical Ethics](#11-limitations-and-clinical-ethics)
12. [Future Work](#12-future-work)
13. [AI Tools Used](#13-ai-tools-used)

---

## 1. Problem Statement: HI-01

**HI-01: Medical Image Diagnostic Assistant** requires developing an assistive deep learning diagnostic system that accurately flags pathologies from medical radiographs, specifically optimized for edge deployment in bandwidth-limited, specialist-deprived rural healthcare facilities.

**TB-Screen AI** directly solves HI-01 for pulmonary tuberculosis (TB) screening:
- Operates **100% offline** on standard dual-core laptop CPUs with **7.19 ms ONNX inference latency**.
- Incorporates deterministic **Out-of-Distribution (OOD) filters** and **radiographic quality diagnostics** to reject non-radiographs or severe blur.
- Employs a **three-tier traffic-light triage system** (Low Risk / Needs Review / High Risk) with clear referral protocols.
- Produces **Grad-CAM pulmonary attention overlays** with an interactive opacity blend slider to show healthcare workers exactly where apical infiltrates lie.
- Generates a **1-click downloadable clinical referral PDF report** with local medical officer sign-off boxes.
- Provides a **high-volume batch screening mode** that automatically ranks mass-screened camp patients by highest risk first.

---

## 2. Team Information

- **Team Name:** Meow Coders
- **Team Members:**
  - **Sayeeda Huriya Fatima, B.Tech** — Team Lead (ML Engineering, Architecture & Deployment Lead)
  - **Syeda Zainab Mahreen, B.Tech** — Team Member (Validation, Clinical UX & Systems Supporter)

---

## 3. Medical Advisor

Dr. Bepar Mohammad Tayyib, MBBS, reviewed the clinical workflow, result messages,
referral guidance, and example outputs of this tool.
He is an advisor and is not a member of the development team.

> [!IMPORTANT]
> **Regulatory & Clinical Trial Disclaimer**: TB-Screen AI is an assistive frontline screening aid designed for preliminary triage in resource-constrained environments. It has not undergone formal prospective clinical trials and does not claim clinical validation. It is not an autonomous diagnostic medical device; all screening outputs must be corroborated with patient clinical history and microbiological confirmation (sputum smear microscopy or GeneXpert MTB/RIF).

---

## 4. Problem Description: Why TB Screening Fails in Rural Clinics

Tuberculosis remains the world's deadliest bacterial infectious killer, causing over **1.3 million deaths annually** worldwide. In India alone, approximately **2.8 million incident cases** occur every year (accounting for 27% of global incidence) with **~340,000 deaths annually (~930 deaths every single day)** (*WHO Global TB Report 2023; Central TB Division 2023*).

While pulmonary TB is completely curable with standard 6-month DOTS antibiotics (rifampicin, isoniazid, pyrazinamide, ethambutol), **frontline screening in rural and under-resourced Primary Health Centers (PHCs) faces severe systemic bottlenecks:**

1. **Radiologist Shortage:** Rural India has an estimated ratio of fewer than **1 radiologist per 100,000 population**. Analog and portable digital X-ray units exist in sub-district centers, but radiographs frequently sit unread for **14 to 28 days** awaiting remote batch interpretation by district hospital specialists.
2. **The Transmission Window:** During this 3-to-4 week diagnostic delay, an untreated patient with active pulmonary cavitations continues aerosolized transmission, infecting an estimated 10 to 15 close contacts per year.
3. **Cartridge Scarcity:** Molecular diagnostics (GeneXpert MTB/RIF) are scarce and expensive (**₹1,500 to ₹2,000 / $18 to $25 per test cartridge**). Indiscriminate testing exhausts rural supplies within days; lack of screening means high-risk patients are missed entirely.
4. **Subclinical Prevalence:** Up to 50% of active cases in community prevalence surveys are smear-negative or minimally symptomatic in early stages, but already display visible pulmonary abnormalities on chest radiographs.

---

## 5. Our Approach & Architectural Pipeline

TB-Screen AI introduces an end-to-end edge pipeline engineered for safety, speed, and frontline usability:

```mermaid
flowchart TD
    A["Raw Radiograph (Upload / Mobile X-ray)"] --> B["Safety & OOD Inspector\n(Chroma, Blank Canvas, Min 100x100)"]
    B -->|Non-CXR / Invalid| B_Err["🚫 Refuse & Display Clear Clinical Alert\n(Zero Raw Python Errors)"]
    B -->|Valid CXR| C["Preprocessing Engine\n- Lung ROI Crop (margin 0.08)\n- CLAHE Contrast (clip 2.0, grid 8x8)\n- ImageNet Normalization (224x224)"]
    C --> D["Inference Engine\n(EfficientNet-B0 / ONNX Runtime 7.19ms)"]
    D --> E["Temperature Scaling Calibration\n(T* = 0.4860 | ECE: 5.10% -> 2.31%)"]
    E --> F{"Three-Tier Risk Triage"}
    F -->|p < 0.15| G["🟢 LOW RISK\nRoutine Primary Care"]
    F -->|0.15 <= p <= 0.60| H["🟡 NEEDS REVIEW\nBorderline Opacity\nMedical Officer & AFB Smear"]
    F -->|p > 0.60| I["🔴 HIGH RISK\nFast-Track Sputum GeneXpert\nInfection Control & DOTS"]
    D --> J["Grad-CAM Saliency Engine\nTarget: Final Conv Block\nInteractive Opacity Slider"]
    G & H & I & J --> K["Clinical Action Card\n+ Live CPU Latency (ms)\n+ Trilingual Interface (EN/HI/TE)"]
    K --> L["1-Click Referral PDF Report\n(Patient ID, Both Images, Doctor Sign-off)"]
    K --> M["Shift Session History & Batch Mode\n(Ranked by Highest Risk First, CSV Export)"]
```

---

## 6. Tech Stack & Libraries

- **Deep Learning Framework:** PyTorch 2.15.0 & Torchvision 0.29.2
- **Optimized Edge Inference:** ONNX 1.23.2 & ONNX Runtime 1.31.1 (CPU Execution Provider)
- **Computer Vision & Contrast:** OpenCV (`opencv-python-headless 5.0.0.93`), Pillow 12.3.0
- **Model Explainability:** Gradient-weighted Class Activation Mapping (`grad-cam 1.5.7`)
- **Scientific Computing & Statistics:** NumPy 2.5.2, Scikit-Learn 1.9.1, SciPy 1.18.1
- **Visualization:** Matplotlib 3.11.1, Seaborn 0.13.2
- **Frontline Clinic UI:** Streamlit 1.66.1
- **Clinical Referral PDF Generation:** `fpdf2 2.9.0`
- **Forensic Dataset Auditing & Testing:** `imagehash 4.3.2`, `pytest 9.2.0`, `openpyxl 3.1.5`

---

## 7. Setup and Installation

### Prerequisites
- Python 3.10+ (Fully compatible up to Python 3.14 on macOS, Linux, and Windows)
- Minimum 2 GB RAM (No GPU required)

```bash
# 1. Clone the repository
git clone https://github.com/meowhuriya/tb-screen-ai.git
cd tb-screen-ai

# 2. Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install strictly pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Acquisition (Optional for Retraining)
The repository contains the deployed model checkpoints (`models/best_model.pth` and `models/model.onnx`), sample radiographs (`app/sample_images/`), and dataset split manifests (`data/processed/splits_manifest.json` and `data/external/montgomery_manifest.json`). The web application and full test suite run immediately without downloading the multi-gigabyte raw datasets.

To retrain the models from scratch or rerun dataset audit scripts:
1. **Primary Dataset (Qatar/Dhaka TB CXR Cohort):**
   - **Source:** [Tuberculosis (TB) Chest X-ray Database on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-x-ray-dataset) (Rahman et al., Qatar University, University of Dhaka, and collaborators).
   - **Download:** Download the archive and extract radiographs into `data/raw/`:
     - `data/raw/Normal/` (3,500 healthy controls)
     - `data/raw/Tuberculosis/` (700 active TB cases)
2. **External Validation Cohort (Montgomery County NIH Dataset):**
   - **Source:** [Montgomery County Department of Health and Human Services & NIH / NLM](https://openi.nlm.nih.gov/) (also hosted on [Kaggle by raddar](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery)).
   - **Download:** Extract the 138 radiographs into:
     - `data/external/montgomery/`
3. **Verify and Generate Splits:**
   ```bash
   python src/dataset.py
   ```

---

## 8. How to Run Locally

### Run the Frontline Streamlit Clinic Application
```bash
streamlit run app/app.py
```
*Opens automatically at `http://localhost:8501`. Features 3 pre-loaded demonstration X-rays for instant zero-upload judge testing.*

### Run the Complete Phase 3 Evaluation & Calibration Suite
```bash
python src/evaluate.py --results_dir results
```
*Generates confusion matrices, ROC/PR curves, temperature calibration curves, bootstrap 95% CIs, and external validation summaries.*

### Export Model to ONNX and Run CPU Benchmark
```bash
python src/export_onnx.py --runs 100
```
*Exports `models/model.onnx` and measures CPU latency across 100 iterations.*

### Run Full Test Suite
```bash
pytest tests/test_pipeline.py -v
```
*Runs all 17 unit tests verifying OOD detection, ONNX inference, input validation (grayscale, RGBA, corrupt payloads), CLAHE preprocessing, input-selection state machine, scanner-shift detection, and PDF generation.*

### Containerized Docker Deployment
```bash
# Build production Docker image (lean multi-stage with .dockerignore)
docker build -t tb-screen-ai .

# Run container on local port 8501
docker run -p 8501:8501 tb-screen-ai
```

### Live Cloud Deployment
- **Official Live Web Application:** [https://tb-screen-ai-hrgejmak9ujcwfm5q4duap.streamlit.app](https://tb-screen-ai-hrgejmak9ujcwfm5q4duap.streamlit.app)
- **GitHub Repository:** [https://github.com/meowhuriya/tb-screen-ai](https://github.com/meowhuriya/tb-screen-ai)

### 1-Click Cloud Deployment Instructions

#### Streamlit Community Cloud:
1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository, branch `main`, and set the entry file path to `app/app.py`.
4. Click **Deploy** — requirements and model checkpoints load automatically in under 60 seconds.

#### Hugging Face Spaces:
1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space) selecting the **Streamlit** SDK.
2. The repository root `README.md` already contains the required YAML header:
   ```yaml
   ---
   title: TB-Screen AI | Frontline Medical Diagnostic Assistant
   emoji: 🫁
   colorFrom: blue
   colorTo: indigo
   sdk: streamlit
   sdk_version: "1.66.1"
   app_file: app/app.py
   pinned: false
   license: mit
   ---
   ```
3. Push the repository to the Space git remote. It spins up automatically without configuration.

---

## 9. Screenshots and Visual Explainability

All visual artifacts are preserved in [`docs/images/`](docs/images/):

### Application Overview & Triage Interface
![TB-Screen AI Application Overview](docs/images/app_overview.png)

### Grad-CAM Saliency Panel Across Diagnostic Presentations
![Grad-CAM Explainability](docs/images/gradcam_examples.png)
*Demonstrates visual localization of active cavitations (True Positives) and clear baseline attention in healthy controls (True Negatives).*

### Forensic Shortcut Learning Audit
![Shortcut Learning Audit](docs/images/shortcut_learning_audit.png)
*Confirms that lung cropping and CLAHE contrast enhancement restrict model attention to internal pulmonary parenchyma rather than peripheral borders or radiopaque letter markers.*

---

## 10. Model Performance & Rigorous Benchmarks

> **Integrity Note:** Every metric, decimal, and confidence interval below is derived directly from real run logs in [`results/metrics.json`](results/metrics.json), [`results/model_comparison.csv`](results/model_comparison.csv), and [`results/cv_5fold_results.json`](results/cv_5fold_results.json).

### 10.1. Untouched In-Domain Test Set Evaluation with 95% Bootstrap CIs
*Evaluated on the untouched sanitized test cohort ($N=627$, $102\text{ TB}, 525\text{ Normal}$, 3 duplicates removed) across 1,000 bootstrap resamples.*  
* **Primary Decision Threshold:** Calibrated Clinical Operating Point $\mathbf{\tau^* = 0.27}$ (balanced frontline triage point)  
* **Community Screening Threshold:** WHO Benchmark $\mathbf{\tau = 0.05}$ (high-sensitivity triage)

| Metric | Point Estimate ($\tau^* = 0.27$) | 95% Bootstrap Confidence Interval | WHO Benchmark ($\tau = 0.05$) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **$0.9780$** | **$[0.9643, 0.9897]$** | $0.9780$ |
| **PR-AUC (Average Precision)** | **$0.9304$** | **$[0.8925, 0.9614]$** | $0.9304$ |
| **Sensitivity (Recall)** | **$88.24\%$** (90/102 detected) | **$[81.58\%, 93.97\%]$** | **$95.10\%$** (97/102 detected) |
| **Specificity** | **$94.67\%$** (497/525 correct) | **$[92.87\%, 96.49\%]$** | **$86.29\%$** (453/525 correct) |
| **Accuracy** | **$93.62\%$** (587/627 correct) | **$[91.55\%, 95.53\%]$** | **$87.72\%$** (550/627 correct) |
| **Precision (PPV)** | **$76.27\%$** | **$[68.57\%, 83.90\%]$** | $57.40\%$ |
| **F1-Score** | **$0.8182$** | **$[0.7605, 0.8710]$** | $0.7159$ |

![Confusion Matrix](docs/images/confusion_matrix.png)
![ROC Curve](docs/images/roc_curve.png)
![Precision-Recall Curve](docs/images/pr_curve.png)

---

### 10.2. Decision Threshold Trade-Off Analysis
*Decision threshold sweep evaluated on the untouched sanitized test cohort ($N=627$):*

| Threshold $\tau$ | Sensitivity (%) | Specificity (%) | Accuracy (%) | Precision (%) | F1-Score | Clinical Interpretation & Triage Role |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\mathbf{0.05}$** | **$\mathbf{95.10\%}$** | **$\mathbf{86.29\%}$** | **$87.72\%$** | $57.40\%$ | $0.7159$ | **Meets WHO Target Benchmark ($\ge 95\%$ Sens)**; 97/102 cases found (5 missed) |
| **$0.10$** | $94.12\%$ | $89.90\%$ | $90.59\%$ | $64.43\%$ | $0.7649$ | High-sensitivity screening cutoff |
| **$0.15$** | $92.16\%$ | $91.81\%$ | $91.87\%$ | $68.61\%$ | $0.7866$ | Intermediate triage cutoff |
| **$0.20$** | $91.18\%$ | $93.14\%$ | $92.82\%$ | $72.09\%$ | $0.8052$ | Balanced rural triage point |
| **$0.25$** | $90.20\%$ | $94.67\%$ | $93.94\%$ | $76.67\%$ | $0.8288$ | Pre-calibrated threshold |
| **$\mathbf{0.27}$** | **$\mathbf{88.24\%}$** | **$\mathbf{94.67\%}$** | **$93.62\%$** | $76.27\%$ | $0.8182$ | **Calibrated Clinical Operating Point ($\tau^*$)** |
| **$0.30$** | $87.25\%$ | $95.24\%$ | $93.94\%$ | $78.07\%$ | $0.8241$ | Conservative screening cutoff |
| **$0.70$** | $80.39\%$ | $98.29\%$ | $95.37\%$ | $90.11\%$ | $0.8497$ | High-specificity conservative cutoff |

---

### 10.3. Probability Calibration (Temperature Scaling)
*Evaluated on the independent validation cohort ($N=630$).*  
Post-hoc Temperature Scaling learned an optimal scalar temperature of **$T^* = 0.4860$**:
- **Brier Score:** Reduced from **$0.0396 \rightarrow 0.0369$** (lower is better).
- **Expected Calibration Error (ECE):** Reduced by more than half from **$5.10\% \rightarrow 2.31\%$**.

![Calibration Reliability Curve](docs/images/calibration_curve.png)
*The reliability curve demonstrates that predicted probabilities closely match empirical positive frequencies.*

---

### 10.4. Multi-Architecture & Ensemble Benchmark
*Evaluated on the untouched sanitized test cohort ($N=627$, $102\text{ TB}, 525\text{ Normal}$, 3 duplicates removed) at standard decision threshold $\mathbf{\tau = 0.50}$; from [`results/model_comparison.csv`](results/model_comparison.csv):*

| Model Pipeline | Parameters | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 (Single)** | 23.51M | **89.15** | **100.00** | **87.05** | **60.00** | **0.7500** | **0.9992** |
| **EfficientNet-B0 (Single)** | **4.01M** | **85.01** | **100.00** | **82.10** | **52.04** | **0.6846** | **0.9962** |
| **Ensemble (EffNet + DenseNet)** | 10.97M | 82.78 | 100.00 | 79.43 | 48.57 | 0.6538 | 0.9986 |
| **DenseNet-121 (Single)** | 6.96M | 79.59 | 99.02 | 75.81 | 44.30 | 0.6121 | 0.9976 |
| **Ensemble + TTA** | 10.97M | 64.91 | 98.04 | 58.48 | 31.45 | 0.4762 | 0.9680 |
| **EfficientNet-B0 + TTA** | 4.01M | 50.08 | 87.25 | 42.86 | 22.88 | 0.3625 | 0.8321 |

![Model Architecture Comparison Chart](docs/images/model_comparison.png)

> [!WARNING]
> **TTA Investigation & Empirical Findings:** In our initial architecture benchmark, Test-Time Augmentation (TTA) showed an unexpected performance collapse to $50.08\%$ accuracy and $42.86\%$ specificity.
> - **Hypothesis:** This collapse was likely caused by a preprocessing discrepancy in the multi-view transform pipeline: the original multi-view TTA routine applied generic resize, crop, and jitter transforms without `CropLungROI(0.08)` and `ApplyCLAHE(2.0)`. Because the convolutional backbone was trained exclusively on localized CLAHE-enhanced, lung-cropped radiographs, passing un-enhanced and un-cropped multi-view variants shifted the input distribution, triggering false-positive over-prediction.
> - **Empirical Verification:** We tested this hypothesis by embedding `CropLungROI(0.08)` and `ApplyCLAHE(2.0)` directly into each TTA view. On the benchmark run (`models/efficientnet_b0_checkpoint.pth`), TTA performance recovered from $50.08\%$ to **$83.41\%$ accuracy**, **$100.00\%$ sensitivity**, and **$0.9975$ ROC-AUC**. Crucially, TTA gave no accuracy gain over single-view inference on the exact same model (**$85.01\%$ accuracy**, **$100.00\%$ sensitivity**, **$0.9962$ ROC-AUC**).
> - **Deployment Decision:** Because TTA gave no accuracy gain over single-view inference ($83.41\%$ vs $85.01\%$) while requiring $3\times$ CPU forward passes per radiograph, single-view inference with CLAHE is retained as the production engine to preserve sub-10ms edge latency on budget clinic hardware.
>
> *Deployment Architecture Choice:* EfficientNet-B0 is selected as the production edge engine because it delivers $100\%$ sensitivity at standard thresholds with $1/6\text{th}$ the parameter count ($4.01\text{M}$ vs $23.51\text{M}$), reducing checkpoint size to $16\text{ MB}$ and enabling $7.19\text{ ms}$ ONNX CPU latency.

---

### 10.5. 5-Fold Stratified Cross-Validation
* **Cohort Evaluated:** Training Cohort only ($N=2,940$ total; 5 folds of $\approx 588$ validation samples each; zero overlap with held-out test cohort).
* **Decision Threshold:** $\mathbf{\tau = 0.27}$ (evaluated on raw uncalibrated validation sigmoid probabilities).
* **Training Recipe:** Shorter 5-epoch training per fold (2 frozen-head epochs + 3 fine-tuning epochs) with focal loss ($\gamma=2.0$, $w_{\text{TB}} \approx 3.0$).

| Metric | Cross-Validation Score (Mean $\pm$ Standard Deviation across 5 Folds) |
| :--- | :---: |
| **Sensitivity (Recall)** | **$99.80\% \pm 0.41\%$** |
| **ROC-AUC** | **$0.9975 \pm 0.0013$** |
| **Accuracy** | $78.77\% \pm 1.97\%$ |
| **Specificity** | $74.57\% \pm 2.33\%$ |
| **Precision** | $44.09\% \pm 2.31\%$ |
| **F1-Score** | $0.6113 \pm 0.0225$ |

*Per-Fold Breakdown ($\tau = 0.27$):*
- **Fold 1:** Acc 76.02% | Sens 100.0% | Spec 71.22% | F1 0.5816 | AUC 0.9991
- **Fold 2:** Acc 81.63% | Sens 100.0% | Spec 77.96% | F1 0.6447 | AUC 0.9983
- **Fold 3:** Acc 79.76% | Sens 100.0% | Spec 75.71% | F1 0.6222 | AUC 0.9979
- **Fold 4:** Acc 79.25% | Sens 100.0% | Spec 75.10% | F1 0.6164 | AUC 0.9966
- **Fold 5:** Acc 77.21% | Sens 98.98% | Spec 72.86% | F1 0.5915 | AUC 0.9955

#### Context on Sensitivity Difference (5-Fold CV: 99.8% vs Untouched Test Set: 88.2% at $\tau = 0.27$):
1. **Shorter Training Recipe & Non-Comparability:** The 5-fold cross-validation evaluated a shorter training recipe (5 total epochs per fold: 2 frozen-head + 3 fine-tuning) on the training cohort ($N=2,940$). Because of this abbreviated schedule, differences in loss weighting, and the use of uncalibrated probabilities, the cross-validation metrics are **not directly comparable** to the final calibrated model.
2. **Loss Weighting & Probability Calibration:** During cross-validation, models were trained with aggressive focal loss minority weighting ($w_{\text{TB}} \approx 3.0$) and evaluated directly on uncalibrated raw sigmoid outputs at $\tau=0.27$. This setup heavily penalized missed TB cases, yielding high sensitivity ($99.80\% \pm 0.41\%$) at the cost of lower specificity ($74.57\% \pm 2.33\%$) and lower precision ($44.09\% \pm 2.31\%$). In contrast, the final deployed checkpoint (`models/best_model.pth`) was trained for 10 epochs and calibrated with post-hoc Temperature Scaling ($T^* = 0.4860$), achieving higher specificity ($94.67\%$) and accuracy ($93.62\%$) on the untouched test set, with calibrated sensitivity at $88.24\%$ ($90/102$ detected).
3. **Catching TB Cases in Practice:** For community screening scenarios that prioritize detecting active cases, operating at the WHO benchmark threshold of $\mathbf{\tau = 0.05}$ achieves **$95.10\%$ sensitivity** ($97/102$ detected, with 5 missed cases) and **$86.29\%$ specificity** ($453/525$ correct) on the untouched test set.

---

### 10.6. External Validation: Montgomery County NIH Dataset
*Completely unseen external cohort from Montgomery County Department of Health and Human Services (Maryland, USA: $N=138$, 80 Normal, 58 Tuberculosis).*  
* **Decision Threshold:** Calibrated Clinical Operating Point $\mathbf{\tau^* = 0.27}$

| Evaluation Cohort | ROC-AUC | Sensitivity (%) | Specificity (%) | Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **In-Domain Sanitized Test Set** | $0.9780$ | $88.24\%$ | $94.67\%$ | $93.62\%$ |
| **External Montgomery County Cohort** | $0.5873$ | $5.17\%$ | $95.00\%$ | $57.25\%$ |
| **Observed Performance Drop** | **$-0.3907$** | **$-83.07\%$** | **$+0.33\%$** | **$-36.37\%$** |

> [!CAUTION]
> **Plain Clinical Finding:** The model does not generalise to a different scanner, and needs local calibration before any real use.
> When evaluated on the Montgomery County NIH cohort, ROC-AUC fell sharply from $0.9780$ to $0.5873$ and sensitivity collapsed to $5.17\%$.
>
> **Root Cause:**
> 1. **Scanner & Digitization Physics:** Montgomery radiographs were captured on an analog Eureka stationary unit and digitized via a flatbed laser film scanner at ultra-high resolution ($4020 \times 4892$) with heavy film-grain noise, whereas training data was captured on digital direct radiography (DR) units with flat contrast curves.
> 2. **Disease Demographics:** Montgomery cases originate from an urban US immigrant screening program dominated by subtle, chronic apical fibrocavitary scarring and calcified granulomas, whereas the training dataset features acute, florid lobar consolidations.
> 3. **Mandatory Protocol:** Any real-world deployment in a new clinic or hospital requires local contrast curve and threshold calibration on that specific facility's X-ray machine before clinical deployment.

---

### 10.7. Grad-CAM Diagnostic Error Analysis
*Analyzed at Calibrated Clinical Operating Point $\mathbf{\tau^* = 0.27}$.*

![Error Analysis Grad-CAM](docs/images/error_analysis.png)

- **Worst False Negatives (e.g. `Tuberculosis-109.png`, $p=0.34\%$):** Missed due to subtle apical micro-nodularity lacking dense confluent consolidation or cavitation. Low radiographic contrast generated under-activated gradient responses.
- **Worst False Positives (e.g. `Normal-2624.png`, $p=98.74\%$):** Normal healthy lungs misclassified due to prominent bronchovascular hilar markings and overlapping clavicular-first rib densities creating high local gradients ($98.7\%$ probability), mimicking early apical infiltration.

---

### 10.8. Hardware Efficiency & Measured Batch Screening Speed
*Evaluated on dual-core CPU with ONNX Runtime (`models/model.onnx`, 0.72 MB) and PyTorch Edge.*

| Performance Benchmark | Target Specification | Real Measured Measurement |
| :--- | :---: | :---: |
| **ONNX Measured Batch Screening Speed** | $> 10\text{ X-rays/sec}$ | **$\mathbf{87.2\text{ X-rays / second}}$** ($\mathbf{11.47\text{ ms / X-ray}}$ end-to-end) |
| **PyTorch + Grad-CAM Batch Speed** | $> 5\text{ X-rays/sec}$ | **$\mathbf{36.9\text{ X-rays / second}}$** ($\mathbf{27.11\text{ ms / X-ray}}$ with full visual explainability) |
| **Median CPU Latency ($p_{50}$)** | $< 2,000\text{ ms}$ (2.0 sec) | **$\mathbf{7.19\text{ ms}}$** (278x faster than target) |
| **95th Percentile ($p_{95}$)** | $< 2,500\text{ ms}$ | **$\mathbf{7.93\text{ ms}}$** |
| **Graph-Only Throughput** | $> 30\text{ X-rays/min}$ | **$\mathbf{8,344.4\text{ X-rays / minute}}$** |
| **Numerical Consistency vs PyTorch** | Prob Diff $< 10^{-4}$ | **$\mathbf{0.00\text{e}+00}$** (Exact match) |
| **RAM Footprint (Peak RSS)** | $< 1,024\text{ MB}$ | **$532.9\text{ MB}$** |

---

## 11. Limitations and Clinical Ethics

1. **Screening Aid, Not Autonomous Diagnosis:** TB-Screen AI is an assistive decision-support tool. It has **not** undergone prospective clinical trials and does not possess regulatory clearance (e.g. FDA, CE-IVD, or CDSCO). All positive screenings require microbiological confirmation via sputum smear microscopy or GeneXpert MTB/RIF cartridge assay.
2. **Concomitant Pulmonary Pathologies:** Bacterial pneumonia, sarcoidosis, histoplasmosis, silicosis, and post-COVID-19 consolidations share radiographic features with pulmonary TB and can induce false positives.
3. **Domain Shift Vulnerability:** As demonstrated in our external validation audit, the model does not generalise to a different scanner, and needs local calibration before any real use.
4. **Pediatric Limitations:** Pediatric pulmonary TB often presents as mediastinal lymphadenopathy rather than apical adult cavitations. This system is validated on adult chest radiographs only.
5. **Color-Tinted Radiographs & Out-of-Distribution Filtering:** The deterministic safety filter inspects chromatic saturation to prevent users from accidentally uploading non-radiographic natural photographs. However, legacy analog radiographs digitized with strong duotone or blue cyanotype color casts (e.g. from historical viewbox backlights) can exceed the chromatic saturation threshold and be rejected as Out-of-Distribution. For example, `sample_tb_moderate.png` (originally `Tuberculosis-418.png`) initially exhibited a strong blue viewbox tint ($34.5\%$ saturation) from its scanner and was converted to standard grayscale RGB so that it is processed cleanly without false OOD rejection. Users operating older film scanners with colored backlights must ensure scans are saved in standard 8-bit grayscale.
6. **Integrity of Demonstration Samples:** All three built-in demonstration samples provided in the application originate strictly from the held-out untouched test split (`splits["test"]["paths"]`), with zero presence in the training cohort:
   - `sample_normal.png` $\rightarrow$ `Normal-698.png` (Test Split)
   - `sample_tb_advanced.png` $\rightarrow$ `Tuberculosis-283.png` (Test Split)
   - `sample_tb_moderate.png` $\rightarrow$ `Tuberculosis-418.png` (Test Split, converted to grayscale)

---

## 12. Future Work

- **Federated Domain Adaptation:** Implement test-time unsupervised domain adaptation (TENT) to normalize contrast curves across diverse rural X-ray generator brands without transmitting patient data.
- **Multi-Class Differential Diagnosis:** Expand classification heads to differentiate active TB from bacterial lobar pneumonia, COVID-19, and pleural effusion.
- **DICOM Integration:** Add native DICOM PACS listener support to interface directly with rural mobile X-ray van digital detectors.
- **Prospective Field Pilot:** Partner with District TB Centers to run a prospective parallel validation study comparing AI triage vs resident radiologist reads.

---

## 13. AI Tools Used

- **Google Antigravity with Gemini 2.5 / Flash:** Model architecture design, modular pipeline refactoring, statistical bootstrap implementations, and forensic auditing scripts.
- **PyTorch & Torchvision:** Core deep learning model definition and transfer learning fine-tuning.
- **PyTorch Grad-CAM:** Saliency map extraction and attention overlay generation.
- **ONNX Runtime:** Graph compilation and CPU latency optimization.
