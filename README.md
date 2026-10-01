# Differential Diagnosis of Monkeypox and Chickenpox Using Vision Transformers and Hybrid Convolutional-Attention Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-0.9%2B-green.svg)](https://github.com/huggingface/pytorch-image-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray.svg)](https://opensource.org/licenses/MIT)

This repository contains the code, data preparation scripts, and evaluation pipelines for our comparative study on 3-class dermatological lesion classification: **Monkeypox**, **Chickenpox**, and **Normal** skin.

The benchmark evaluates 12 model architectures—spanning pure vision transformers, hierarchical transformers, and hybrid CNN-transformer networks—alongside explainability methods (Attention Rollout, Grad-CAM, Integrated Gradients) and an interactive diagnostic evaluation tool.

---

## Overview

Cutaneous lesions caused by Monkeypox virus (MPXV) often present similarly to Varicella-Zoster virus (Chickenpox), particularly during vesiculopustular stages. In low-resource and outpatient clinical settings, distinguishing these conditions visually is difficult.

This project investigates whether recent transformer-based vision architectures—which model non-local context differently than classical CNNs—can reliably differentiate between these conditions. We evaluate models across standard single-crop inference and test-time augmentation (TTA), using an audited multi-source dataset designed to eliminate cross-split patient and crop-level leakage.

### Evaluated Model Architectures

The benchmark includes 12 architectures implemented via `timm` and fine-tuned under a standardized protocol:

| Architecture | Model Variant | Model Family | Parameters | Primary Reference |
| :--- | :--- | :--- | :--- | :--- |
| **ViT** | `vit_base_patch16_224` | Pure Transformer | 86.6M | Dosovitskiy et al., ICLR 2021 |
| **DeiT** | `deit_base_patch16_224` | Distilled Transformer | 86.6M | Touvron et al., ICML 2021 |
| **Swin** | `swin_base_patch4_window7_224` | Hierarchical / Local Window | 87.8M | Liu et al., ICCV 2021 |
| **BEiT** | `beit_base_patch16_224` | Masked Pre-trained | 86.5M | Bao et al., ICLR 2022 |
| **MaxViT** | `maxvit_base_tf_224` | Multi-Axis Attention | 119.5M | Tu et al., ECCV 2022 |
| **PiT** | `pit_b_224` | Spatial Pooling ViT | 73.8M | Heo et al., NeurIPS 2021 |
| **MAE** | `vit_base_patch16_224.mae` | Self-Supervised ViT | 86.6M | He et al., CVPR 2022 |
| **FlexiViT** | `flexivit_base.patch16_in21k` | Flexible Patch Sizes | 86.6M | Beyer et al., CVPR 2023 |
| **CoAtNet** | `coatnet_1_rw_224.sw_in1k` | Hybrid (MBConv + Attention) | 42.2M | Dai et al., NeurIPS 2021 |
| **CvT** | `cvt_13` | Hybrid (Conv Token Projections) | 20.0M | Wu et al., ICCV 2021 |
| **Visformer** | `visformer_small` | Hybrid (Conv-Attention Blocks) | 40.2M | Chen et al., ICCV 2021 |
| **ConViT** | `convit_small` | Hybrid (Gated Positional Self-Attention) | 27.8M | d'Ascoli et al., ICML 2021 |

---

## Training Methodology

All models follow a unified training configuration to enable fair comparisons:

1. **Input Normalization & Grayscale Channel Adaptation:**
   Lesion images are processed at $224 \times 224$ pixels. Single-channel grayscale inputs are repeated across 3 channels to preserve compatibility with ImageNet-pretrained positional embeddings and patch projection weights.

2. **Domain-Motivated Augmentation:**
   Training applies horizontal and vertical flips ($p=0.5$), random rotations ($\pm 15^\circ$), affine scaling ($0.95 \text{ to } 1.05$) with translation ($\pm 5\%$), slight illumination jitter ($\pm 10\%$), and small-area random erasing ($p=0.15$, area $0.02\text{--}0.08$).

3. **Optimization:**
   - Optimizer: AdamW with weight decay $0.01$
   - Learning Rates: Differential scheme with backbone at $2 \times 10^{-5}$ and classification head at $2 \times 10^{-4}$
   - Schedule: 3-epoch linear warmup followed by cosine annealing decay
   - Loss Function: Cross-Entropy with label smoothing ($0.1$)
   - Early Stopping: Monitored on validation macro-F1 (patience of 7 epochs after warmup)

4. **Inference Protocols:**
   Results are reported under both:
   - **Standard Inference:** Single-pass evaluation on original test images.
   - **Test-Time Augmentation (TTA):** Softmax probability averaging over original and horizontal flip transforms.

---

## Dataset Curation and Audit

Raw images were integrated from multiple publicly available dermatological repositories and subjected to an automated deduplication and leakage-auditing pipeline:

- **Deduplication:** Full SHA-256 exact matching and perceptual hashing (pHash, Hamming distance $\le 4$) were run across all source folders to remove duplicate files and multi-fold redundancies.
- **Leakage Prevention:** Near-duplicate clusters and multiple crops/augmentations originating from the same lesion/patient were identified and constrained to the same split. No image variants cross the train/test boundary.
- **Normal Class Standardization:** Outdated non-clinical portraits were replaced with verified healthy skin close-up patches from the Jakariya Nayeem dataset.

### Distribution

| Class | Train Set | Test Set (Held-Out) | Total Images | Class Proportion |
| :--- | :---: | :---: | :---: | :---: |
| **Chickenpox** | 406 | 105 | 511 | 33.27% |
| **Monkeypox** | 408 | 105 | 513 | 33.40% |
| **Normal** | 407 | 105 | 512 | 33.33% |
| **Total** | **1,221** | **315** | **1,536** | **100.00%** |

Validation splits (15% of the training pool, 183 images) are partitioned dynamically via `StratifiedShuffleSplit` inside each notebook, keeping the held-out test set completely untouched until evaluation.

Complete audit manifests and inventory tables are in [`reports/`](./reports).

---

## Interactive Prediction Tool

For quick clinical review and sample testing, the repository provides both a script and an interactive notebook interface:

- **CLI Tool:** [`predict_and_evaluate.py`](./predict_and_evaluate.py)
- **Interactive Notebook:** [`15_Interactive_Model_Inspector.ipynb`](./15_Interactive_Model_Inspector.ipynb)

### Features
- Select any of the 12 trained architectures or supply a custom checkpoint path.
- Provide custom input images (single image, image list, or a folder) or draw $N$ random samples from the held-out test directory.
- Generate visual diagnostic cards showing the predicted label, confidence percentage, true label (when known), and class-wise probability distributions.
- Compute sample metrics (Accuracy, Balanced Accuracy, Macro-F1, Precision, Recall, Confusion Matrix) when evaluating ground-truth samples.

![Sample Diagnostic Output](./MaxViT_Test_Random.png)

### CLI Usage Examples

```bash
# Evaluate 6 random test images using CoAtNet with TTA
python predict_and_evaluate.py --model 11_CoAtNet_Hybrid --random-test-samples 6 --tta

# Run inference on custom image files
python predict_and_evaluate.py --model 01_ViT_Base --custom-images sample_lesion1.jpg sample_lesion2.jpg

# Evaluate using a specific checkpoint file
python predict_and_evaluate.py --checkpoint results/05_MaxViT_Base/checkpoints/best_model.pth --random-test-samples 10
```

---

## Explainability (XAI)

Notebook [`09_Explainable_AI_XAI.ipynb`](./09_Explainable_AI_XAI.ipynb) provides qualitative attribution analysis:

1. **Attention Rollout:** Recursively multiplies attention matrices across transformer layers to visualize global information flow to the `[CLS]` token.
2. **ViT Grad-CAM:** Generates activation heatmaps from gradients backpropagated to the final attention blocks.
3. **Integrated Gradients:** Computes path integrals of feature gradients with respect to a blank baseline to identify individual pixel-level contributions.

---

## Repository Structure

```text
.
├── 01_Vision_Transformer_ViT.ipynb         # ViT-Base training & evaluation
├── 02_DeiT_Classification.ipynb            # DeiT training & evaluation
├── 03_Swin_Transformer.ipynb               # Swin Transformer training & evaluation
├── 04_BEiT_Classification.ipynb            # BEiT training & evaluation
├── 05_MaxViT_Classification.ipynb          # MaxViT training & evaluation
├── 06_PiT_Classification.ipynb             # PiT training & evaluation
├── 07_MAE_Classification.ipynb             # MAE training & evaluation
├── 08_FlexiViT_Classification.ipynb        # FlexiViT training & evaluation
├── 09_Explainable_AI_XAI.ipynb             # Interpretability suite (Rollout, Grad-CAM)
├── 10_Model_Comparison_and_Paper_Figures.ipynb # Master cross-model synthesis & LaTeX tables
├── 11_CoAtNet_Hybrid.ipynb                 # CoAtNet training & evaluation
├── 12_CvT_Hybrid.ipynb                     # CvT training & evaluation
├── 13_Visformer_Hybrid.ipynb               # Visformer training & evaluation
├── 14_ConViT_Hybrid.ipynb                  # ConViT training & evaluation
├── 15_Interactive_Model_Inspector.ipynb    # Interactive sample evaluation notebook
├── predict_and_evaluate.py                 # Standalone prediction and evaluation CLI
├── requirements.txt                        # Python dependencies
├── reports/                                # Audit manifests, distribution tables, figures
│   ├── balancing_report.csv
│   ├── class_distribution.csv
│   ├── dataset_audit_report.md
│   ├── split_manifest.csv
│   └── figures/
│       └── 01_class_distribution_comparison.png
├── Custom_Test/                            # Sample test images for demonstration
└── Monkeypox_Transformer_Research_Paper_DRAFT1.pdf # Draft research manuscript
```

---

## Getting Started

### Google Colab

The notebooks are designed to run in Google Colab with GPU hardware:

1. In your Google Drive root, place your dataset in a folder named `Monkeypox_Final/`:
   ```text
   MyDrive/
   └── Monkeypox_Final/
       └── mkpox_dataset_grayscale/   # (or processed/)
           ├── train/
           │   ├── Chickenpox/
           │   ├── Monkeypox/
           │   └── Normal/
           └── test/
               ├── Chickenpox/
               ├── Monkeypox/
               └── Normal/
   ```
2. Open the desired notebook in Colab.
3. Select a GPU runtime (**Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU**).
4. Run the notebook cells sequentially. Each notebook automatically syncs images to local SSD storage (`/content/dataset_local`) during the first run to accelerate training throughput, while output checkpoints, tables, and figures are written back to Google Drive.

### Local Setup

To set up the environment locally:

```bash
git clone https://github.com/Neamul09/Monkeypox-Detection-Vision-Transformers.git
cd Monkeypox-Detection-Vision-Transformers

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Citation

If you use this benchmark, models, or dataset methodology in your research, please cite:

```bibtex
@article{morshed2024monkeypox,
  title={Differential Diagnosis of Monkeypox and Varicella-Zoster Using Vision Transformers and Hybrid Convolutional-Attention Networks with Explainable AI},
  author={Morshed, Md. Neamul and collaborators},
  journal={Working Paper},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
