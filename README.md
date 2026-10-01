# 3-Class Monkeypox Detection Using Vision Transformers & Hybrid CNN-Transformers with Explainable AI (XAI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-0.9%2B-orange.svg)](https://github.com/huggingface/pytorch-image-models)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade deep learning framework for the automated differential diagnosis of **Monkeypox**, **Chickenpox**, and **Normal** skin lesions using **12 state-of-the-art Vision Transformer and Hybrid CNN-Transformer architectures**, accompanied by **Explainable AI (XAI)**, rigorous dataset audit pipelines, and an **interactive clinical prediction and inspection system**.

---

## 📑 Table of Contents
- [Clinical Background & Objectives](#-clinical-background--objectives)
- [Evaluated Architectures (12 Models)](#-evaluated-architectures-12-models)
- [Key Engineering & Methodological Highlights](#-key-engineering--methodological-highlights)
- [Interactive Clinical Prediction & Diagnostic System](#-interactive-clinical-prediction--diagnostic-system)
- [Explainable AI (XAI) Suite](#-explainable-ai-xai-suite)
- [Research-Grade Dataset Harmonization & Audit](#-research-grade-dataset-harmonization--audit)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
  - [Running on Google Colab (Recommended)](#running-on-google-colab-recommended)
  - [Running Locally / CLI](#running-locally--cli)
- [Citation](#-citation)
- [License](#-license)

---

## 🩺 Clinical Background & Objectives

Monkeypox (Mpox) presents with cutaneous eruptions, vesiculopustular lesions, and umbilicated papules that closely mimic other dermatological conditions, most notably Chickenpox (Varicella-Zoster Virus). Differentiating these clinically during early presentation is challenging, leading to misdiagnoses or delayed containment.

This framework provides an end-to-end reproducible research ecosystem that benchmarks 12 distinct neural architectures to identify optimal trade-offs between inductive bias, global self-attention, parameter count, and diagnostic sensitivity.

```
                         Input Dermatological Image
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    Pure / Hierarchical ViTs               Hybrid CNN-Transformers
    (ViT, DeiT, Swin, BEiT,                (CoAtNet, CvT, Visformer,
     MaxViT, PiT, MAE, FlexiViT)                     ConViT)
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                      Ensemble Test-Time Augmentation
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
     Diagnostic Prediction Card                 Explainability (XAI)
    - True vs. Predicted Label             - Attention Rollout
    - Confidence Percentage (%)            - ViT Grad-CAM
    - 3-Class Probability Bars             - Integrated Gradients
```

---

## 🔬 Evaluated Architectures (12 Models)

The framework benchmarks **8 Vision Transformers** and **4 Hybrid CNN-Transformers** under an identical training, validation, and empirical Test-Time Augmentation (TTA) protocol:

| # | Model | Family | `timm` Identifier | Key Architectural Innovation | Paper Reference |
| :-: | :--- | :--- | :--- | :--- | :--- |
| `01` | **ViT-Base** | Pure ViT | `vit_base_patch16_224` | Foundational patch-based global self-attention | Dosovitskiy et al. (ICLR 2021) |
| `02` | **DeiT** | Distilled ViT | `deit_base_patch16_224` | Data-efficient training with token-based distillation | Touvron et al. (ICML 2021) |
| `03` | **Swin Transformer** | Hierarchical | `swin_base_patch4_window7_224` | Shifted window self-attention with linear computational complexity | Liu et al. (ICCV 2021) |
| `04` | **BEiT** | Masked Pre-trained | `beit_base_patch16_224` | Self-supervised masked image modeling with visual discrete VAE tokens | Bao et al. (ICLR 2022) |
| `05` | **MaxViT** | Multi-Axis | `maxvit_base_tf_224` | Multi-axis attention combining local block self-attention and sparse global grid attention | Tu et al. (ECCV 2022) |
| `06` | **PiT** | Pooling ViT | `pit_b_224` | Channel-wise depth pooling between transformer stages | Heo et al. (NeurIPS 2021) |
| `07` | **MAE** | Autoencoder | `vit_base_patch16_224.mae` | High-capacity masked autoencoder representation fine-tuning | He et al. (CVPR 2022) |
| `08` | **FlexiViT** | Dynamic Patch | `flexivit_base.patch16_in21k` | Flexible multi-scale patch projection matrices | Beyer et al. (CVPR 2023) |
| `11` | **CoAtNet** | Hybrid | `coatnet_1_rw_224.sw_in1k` | MBConv depthwise separable convs in stages 1–2 + relative attention in stages 3–4 | Dai et al. (NeurIPS 2021) |
| `12` | **CvT** | Hybrid | `cvt_13` | Replaces linear token projections with depthwise conv projections; overlapping downsampling | Wu et al. (ICCV 2021) |
| `13` | **Visformer** | Hybrid | `visformer_small` | Embeds depthwise convolutions and Squeeze-and-Excitation (SE) blocks directly in attention blocks | Chen et al. (ICCV 2021) |
| `14` | **ConViT** | Hybrid | `convit_small` | Gated Positional Self-Attention (GPSA), learning to transition from convolutional to global attention | d'Ascoli et al. (ICML 2021) |

---

## ⚡ Key Engineering & Methodological Highlights

1. **10x–20x Training Acceleration via Colab NVMe Caching:**
   - Automatically syncs dataset files to the instance's high-speed local SSD (`/content/dataset_local`).
   - Completely eliminates Google Drive FUSE latency, reducing per-epoch training time from minutes to 10–20 seconds.
2. **Differential Learning Rates:**
   - Backbone parameters fine-tune at $2\times 10^{-5}$ to preserve ImageNet representations.
   - Classification heads train at $2\times 10^{-4}$ for rapid adaptation to lesion features.
3. **Domain-Informed Dermatological Augmentations:**
   - Horizontal & vertical flips ($p=0.5$), slight rotations ($\pm 15^\circ$), affine scaling/translation ($\pm 5\%$).
   - Illumination/contrast jitter ($\pm 10\%$).
   - Constrained patch erasing ($p=0.15$, scale $0.02\text{--}0.08$) to prevent occluding entire small umbilicated lesions.
4. **Macro-F1 Monitored Early Stopping:**
   - Monitors **Validation Macro-F1** with `PATIENCE = 7` to give equal clinical weight to all 3 classes regardless of frequency.
5. **Empirical Test-Time Augmentation (TTA):**
   - Evaluates held-out test data under both standard single-pass and TTA (horizontal flip ensemble), reporting exact empirical differences ($\Delta$).
6. **Publication Figures (300 DPI) & LaTeX Tables:**
   - Generates loss/accuracy/F1 convergence curves, raw and normalized confusion matrices, multi-class One-vs-Rest ROC curves, and Precision-Recall (PR) curves automatically saved directly in LaTeX `\begin{table}` and 300 DPI PNG formats.

---

## 🖥️ Interactive Clinical Prediction & Diagnostic System

A dedicated, self-contained diagnostic application is provided in both interactive notebook form ([`15_Interactive_Model_Inspector.ipynb`](./15_Interactive_Model_Inspector.ipynb)) and standalone CLI form ([`predict_and_evaluate.py`](./predict_and_evaluate.py)):

### Capabilities:
- **Model Selection:** Choose from any of the 12 models via a dropdown menu or provide a custom `.pth` checkpoint.
- **Dual Input Modes:**
  - **Custom Images:** Predict on 1 or more user-supplied lesion photographs or a folder of images.
  - **Random Test Sampling:** Sample $N$ random images ($1\dots24$) from the held-out test split.
- **Diagnostic Cards:**
  - Overlays the **Predicted Label and Confidence (%)** directly on the image.
  - Color-coded indicator: 🟢 **Green** (Correct), 🔴 **Red** (Misclassified), 🔵 **Blue** (Unlabeled).
  - Displays a side-by-side **horizontal probability bar chart** showing individual probabilities for `Chickenpox`, `Monkeypox`, and `Normal`.
- **Quantitative Metrics (when ground truth is available):**
  - Accuracy, Balanced Accuracy, Macro-F1, Precision, Recall/Sensitivity, Specificity, and an annotated Confusion Matrix heatmap.

### Example Visual Output:
```text
┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│ [🟢 PRED: Monkeypox (94.2%)]    │ │ Probability Breakdown:          │
│ True: Monkeypox                 │ │                                 │
│                                 │ │ Chickenpox: [===]        4.1%   │
│       [ Lesion Image ]          │ │ Monkeypox:  [==========] 94.2%  │
│                                 │ │ Normal:     [=]          1.7%   │
└─────────────────────────────────┘ └─────────────────────────────────┘
```

```bash
# Run via CLI on 6 random test images with CoAtNet and TTA
python predict_and_evaluate.py --model 11_CoAtNet_Hybrid --random-test-samples 6 --tta

# Run on custom clinical photographs
python predict_and_evaluate.py --model 01_ViT_Base --custom-images sample1.jpg sample2.jpg
```

---

## 🧠 Explainable AI (XAI) Suite

Notebook [`09_Explainable_AI_XAI.ipynb`](./09_Explainable_AI_XAI.ipynb) provides clinical interpretability to verify that model predictions are guided by genuine pathogenic lesion morphology rather than background image artifacts:
- **Attention Rollout:** Propagates attention weights across all transformer heads and layers to highlight spatial regions dominating classification.
- **Vision Transformer Grad-CAM:** Computes class-discriminative gradients with respect to the final transformer layer's token embeddings.
- **Integrated Gradients (Captum):** Computes pixel-level attribution baselines to map individual feature contributions.

---

## 📊 Research-Grade Dataset Harmonization & Audit

The dataset preparation pipeline harmonized **7 independent raw medical image datasets** into an audited, balanced 3-class distribution with **zero train/test data leakage**:

### Audit Checklist & Integrity Metrics:
- **Raw Data Strictly Read-Only:** All raw source datasets in `DATASET/` remain untouched.
- **Zero Exact Duplicates:** 0 duplicate SHA-256 hashes across splits.
- **Zero Cross-Split Leakage:** 0 perceptual near-duplicates ($\text{pHash distance} \le 4$) between train and test.
- **Zero Subject/Lesion ID Overlap:** Distinct lesion and subject identifiers across splits.
- **Normal Class Standardization:** Replaced low-quality non-clinical images with verified dermatological close-up patches from `jakariyanayeem/Healthy`.

### Final Dataset Distribution:
```text
processed/
├── train/
│   ├── Chickenpox/   (406 images)
│   ├── Monkeypox/    (408 images)
│   └── Normal/       (407 images)   ──► Total: 1,221 images (79.5%)
└── test/
    ├── Chickenpox/   (105 images)
    ├── Monkeypox/    (105 images)
    └── Normal/       (105 images)   ──► Total:   315 images (20.5%)
                                     ──────────────────────────────
                                     GRAND TOTAL: 1,536 images
```

All audit logs, inventory tables, and balancing reports are documented in the [`reports/`](./reports) directory.

---

## 📁 Repository Structure

```text
├── 01_Vision_Transformer_ViT.ipynb         # ViT-Base training & evaluation
├── 02_DeiT_Classification.ipynb            # DeiT training & evaluation
├── 03_Swin_Transformer.ipynb               # Swin Transformer training & evaluation
├── 04_BEiT_Classification.ipynb            # BEiT training & evaluation
├── 05_MaxViT_Classification.ipynb          # MaxViT training & evaluation
├── 06_PiT_Classification.ipynb             # PiT training & evaluation
├── 07_MAE_Classification.ipynb             # MAE training & evaluation
├── 08_FlexiViT_Classification.ipynb        # FlexiViT training & evaluation
├── 09_Explainable_AI_XAI.ipynb             # Attention Rollout & Grad-CAM
├── 10_Model_Comparison_and_Paper_Figures.ipynb # Master 12-model synthesis & LaTeX tables
├── 11_CoAtNet_Hybrid.ipynb                 # CoAtNet Hybrid CNN-Transformer
├── 12_CvT_Hybrid.ipynb                     # CvT Hybrid CNN-Transformer
├── 13_Visformer_Hybrid.ipynb               # Visformer Hybrid CNN-Transformer
├── 14_ConViT_Hybrid.ipynb                  # ConViT Hybrid CNN-Transformer
├── 15_Interactive_Model_Inspector.ipynb    # Interactive diagnostic inspection tool
├── predict_and_evaluate.py                 # Standalone prediction & evaluation CLI
├── requirements.txt                        # Python dependencies
├── reports/                                # Audit reports, CSV manifests & figures
│   ├── balancing_report.csv
│   ├── class_distribution.csv
│   ├── dataset_audit_report.md
│   ├── split_manifest.csv
│   └── figures/
│       └── 01_class_distribution_comparison.png
└── Monkeypox_Transformer_Research_Paper_DRAFT1.pdf # Draft research manuscript
```

---

## 🚀 Quickstart Guide

### Running on Google Colab (Recommended)
1. In your Google Drive root (`MyDrive`), ensure your project folder is named:
   ```text
   MyDrive/Monkeypox_Final/
   ```
2. Upload your dataset folder (`mkpox_dataset_grayscale` or `processed`) containing `train/` and `test/` subfolders (`Chickenpox/`, `Monkeypox/`, `Normal/`).
3. Open any notebook (`01` through `15`) in Google Colab and set Runtime to **GPU** (T4 or higher).
4. Run all cells. Outputs (checkpoints, figures, tables, and JSON summaries) are saved automatically to `MyDrive/Monkeypox_Final/results/`.

### Running Locally / CLI
1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/Neamul09/Monkeypox-Detection-Vision-Transformers.git
   cd Monkeypox-Detection-Vision-Transformers
   pip install -r requirements.txt
   ```
2. Run interactive model inspection from the terminal:
   ```bash
   python predict_and_evaluate.py --model 11_CoAtNet_Hybrid --random-test-samples 6 --tta
   ```

---

## 📖 Citation

If you use this codebase, models, or audit methodology in your research, please cite:

```bibtex
@article{morshed2024monkeypox,
  title={Differential Diagnosis of Monkeypox and Varicella-Zoster Using Vision Transformers and Hybrid Convolutional-Attention Networks with Explainable AI},
  author={Morshed, Md. Neamul and collaborators},
  journal={Preprint / Working Paper},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
