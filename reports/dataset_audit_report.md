# Final Research-Grade Dataset Audit & Preparation Report

**Project:** 3-Class Dermatological Diagnosis (`Chickenpox`, `Monkeypox`, `Normal`)  
**Date:** 2026-09-22 21:54:24  
**Target Pipeline:** Vision Transformer (`01_Vision_Transformer_ViT.ipynb`)  
**Processed Dataset Location:** `processed/` (with `train/` and `test/` subdirectories)  
**Reports Location:** `reports/`  

---

## 1. Executive Summary & Verification Checklist

- [x] **Exactly 3 Canonical Classes:** `Chickenpox`, `Monkeypox`, `Normal`.
- [x] **Zero Exact Duplicates:** 0 exact duplicate files across the entire dataset.
- [x] **Zero Train/Test Leakage:** 0 near-duplicate pairs (pHash distance $\le 4$) and 0 patient/lesion prefix overlaps between `train` and `test`.
- [x] **Physical Directory Structure:** Exactly `train/` and `test/` exist. **NO** physical `val/` or `validation/` directory exists.
- [x] **Perfect Test Balance:** Exactly **105 Chickenpox, 105 Monkeypox, 105 Normal** (1:1:1 ratio, 315 total).
- [x] **Balanced Train Split:** **406 Chickenpox, 408 Monkeypox, 407 Normal** (1,221 total).
- [x] **Total Images:** **1,536 images** (79.5% Train / 20.5% Test).
- [x] **ViT Compatibility:** Directly compatible with `datasets.ImageFolder` in `01_Vision_Transformer_ViT.ipynb`.
- [x] **Full Traceability:** Every image is uniquely identified and documented in `split_manifest.csv`.

---

## 2. Final Physical Dataset Distribution

```text
processed/
├── train/
│   ├── Chickenpox/   (406 images)
│   ├── Monkeypox/    (408 images)
│   └── Normal/       (407 images)
└── test/
    ├── Chickenpox/   (105 images)
    ├── Monkeypox/    (105 images)
    └── Normal/       (105 images)
```

| Canonical Class | Train Count | Test Count | Total Count | Class Proportion (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Chickenpox** | 406 | 105 | 511 | 33.27% |
| **Monkeypox** | 408 | 105 | 513 | 33.40% |
| **Normal** | 407 | 105 | 512 | 33.33% |
| **TOTAL** | **1,221** | **315** | **1,536** | **100.00%** |
