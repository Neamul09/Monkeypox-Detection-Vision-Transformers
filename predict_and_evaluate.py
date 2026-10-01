import os
import sys
import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
from torchvision import transforms

# Set seed for reproducible sampling
random.seed(42)
np.random.seed(42)

# Default class configuration
CANONICAL_CLASSES = ["Chickenpox", "Monkeypox", "Normal"]
IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# Evaluated models mapping
AVAILABLE_MODELS = {
    "1": ("01_ViT_Base", "vit_base_patch16_224"),
    "2": ("02_DeiT_Base", "deit_base_patch16_224"),
    "3": ("03_Swin_Base", "swin_base_patch4_window7_224"),
    "4": ("04_BEiT_Base", "beit_base_patch16_224"),
    "5": ("05_MaxViT_Base", "maxvit_base_tf_224"),
    "6": ("06_PiT_Base", "pit_b_224"),
    "7": ("07_MAE_Base", "vit_base_patch16_224.mae"),
    "8": ("08_FlexiViT_Base", "flexivit_base.patch16_in21k"),
    "9": ("11_CoAtNet_Hybrid", "coatnet_1_rw_224.sw_in1k"),
    "10": ("12_CvT_Hybrid", "cvt_13"),
    "11": ("13_Visformer_Hybrid", "visformer_small"),
    "12": ("14_ConViT_Hybrid", "convit_small"),
}

# Image evaluation transform
eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

class MonkeypoxPredictor:
    """
    Unified Inference & Evaluation Engine for Dermatological Image Classification.
    Supports ViT, DeiT, Swin, BEiT, MaxViT, PiT, MAE, FlexiViT, CoAtNet, CvT, Visformer, ConViT.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model_name = model_name
        self.class_names = CANONICAL_CLASSES
        self.model = None
        self.model_identifier = None
        
        self._load_model()

    def _load_model(self):
        """Loads model weights from checkpoint or instantiates architecture."""
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        checkpoint_dict = None
        if self.checkpoint_path and self.checkpoint_path.exists():
            print(f"Loading checkpoint: {self.checkpoint_path}...")
            checkpoint_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if 'class_names' in checkpoint_dict:
                self.class_names = checkpoint_dict['class_names']
            if 'config' in checkpoint_dict and 'model_identifier' in checkpoint_dict['config']:
                self.model_identifier = checkpoint_dict['config']['model_identifier']

        if not self.model_identifier:
            if self.model_name and self.model_name in AVAILABLE_MODELS:
                self.model_identifier = AVAILABLE_MODELS[self.model_name][1]
            else:
                self.model_identifier = "vit_base_patch16_224"

        print(f"Instantiating model architecture: {self.model_identifier} (Classes: {self.class_names})...")
        try:
            self.model = timm.create_model(self.model_identifier, pretrained=False, num_classes=len(self.class_names))
        except Exception as e:
            print(f"Direct creation failed ({e}). Trying fallback...")
            self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(self.class_names))

        if checkpoint_dict and 'model_state_dict' in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
            print(f"Successfully loaded checkpoint weights (Epoch {checkpoint_dict.get('epoch', 'N/A')})!")
        else:
            print("Note: Running with initialized model weights (no trained checkpoint provided).")

        self.model.to(self.device)
        self.model.eval()

    def predict_single(self, image_path: Path, use_tta: bool = False) -> Dict[str, Any]:
        """Predicts single image with probabilities."""
        with Image.open(image_path) as raw_img:
            raw_img_rgb = raw_img.convert('RGB')
            tensor_img = eval_transforms(raw_img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_orig = self.model(tensor_img)
            p_orig = torch.softmax(out_orig, dim=1)

            if use_tta:
                tensor_flipped = torch.flip(tensor_img, dims=[-1])
                out_flip = self.model(tensor_flipped)
                p_flip = torch.softmax(out_flip, dim=1)
                probs = (0.5 * (p_orig + p_flip)).cpu().numpy()[0]
            else:
                probs = p_orig.cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_idx]
        conf = float(probs[pred_idx])

        # Infer true label if image is inside a class folder
        true_label = None
        parent_name = image_path.parent.name
        if parent_name in self.class_names:
            true_label = parent_name
        elif any(c.lower() in image_path.stem.lower() for c in self.class_names):
            for c in self.class_names:
                if c.lower() in image_path.stem.lower():
                    true_label = c
                    break

        return {
            'image_path': str(image_path),
            'filename': image_path.name,
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': conf,
            'is_correct': (true_label == pred_label) if true_label else None,
            'probabilities': {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        }

    def predict_batch(self, image_paths: List[Path], use_tta: bool = False) -> List[Dict[str, Any]]:
        """Batch prediction over multiple images."""
        results = []
        for p in image_paths:
            res = self.predict_single(p, use_tta=use_tta)
            results.append(res)
        return results


def sample_random_test_images(test_dir: Path, num_samples: int = 6) -> List[Path]:
    """Selects N random test images across all classes in test_dir."""
    all_test_images = []
    for c in CANONICAL_CLASSES:
        cdir = test_dir / c
        if cdir.exists():
            files = [f for f in cdir.glob('*') if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            all_test_images.extend(files)

    if not all_test_images:
        for f in test_dir.rglob('*'):
            if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                all_test_images.append(f)

    if not all_test_images:
        raise FileNotFoundError(f"No test images found in: {test_dir}")

    num_samples = min(num_samples, len(all_test_images))
    return random.sample(all_test_images, num_samples)


def render_prediction_visuals(
    predictions: List[Dict[str, Any]],
    output_path: Path,
    title_suffix: str = ""
):
    """
    Renders clean, publication-quality visual cards showing:
    1. The dermatological image with predicted and true labels overlaid directly.
    2. A side-by-side probability bar chart for all 3 classes.
    """
    n = len(predictions)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(6.5 * cols, 4.5 * rows), dpi=200)
    plt.suptitle(f"Model Inference & Probability Analysis {title_suffix}", fontsize=16, y=0.99)

    for idx, p in enumerate(predictions):
        img_p = Path(p['image_path'])
        true_lbl = p['true_label']
        pred_lbl = p['predicted_label']
        conf = p['confidence'] * 100.0
        probs = p['probabilities']
        is_corr = p['is_correct']

        # Determine card border / badge color
        if is_corr is True:
            badge_color = "#2ca02c" # Green
            status_text = "CORRECT"
        elif is_corr is False:
            badge_color = "#d62728" # Red
            status_text = "INCORRECT"
        else:
            badge_color = "#1f77b4" # Blue (unlabeled)
            status_text = "PREDICTED"

        # 1. Image Subplot
        ax_img = plt.subplot2grid((rows, cols * 2), (idx // cols, (idx % cols) * 2))
        try:
            with Image.open(img_p) as img:
                ax_img.imshow(img.convert('RGB'))
        except Exception:
            ax_img.text(0.5, 0.5, "Image Load Error", ha='center', va='center')

        ax_img.axis('off')

        # Annotation Badge directly above/on the image
        badge_str = f"Pred: {pred_lbl} ({conf:.1f}%)\nTrue: {true_lbl if true_lbl else 'Unknown'}"
        ax_img.set_title(
            badge_str,
            fontsize=11,
            fontweight='bold',
            color='white',
            backgroundcolor=badge_color,
            pad=6
        )

        # 2. Probability Bar Chart Subplot
        ax_bar = plt.subplot2grid((rows, cols * 2), (idx // cols, (idx % cols) * 2 + 1))
        classes = list(probs.keys())
        prob_vals = [probs[c] * 100.0 for c in classes]
        bar_colors = ['#2ca02c' if c == pred_lbl else '#aec7e8' for c in classes]

        bars = ax_bar.barh(classes, prob_vals, color=bar_colors, edgecolor='black', height=0.55)
        ax_bar.set_xlim(0, 105)
        ax_bar.set_xlabel("Probability (%)", fontsize=10)
        ax_bar.grid(axis='x', linestyle='--', alpha=0.5)

        for bar in bars:
            w = bar.get_width()
            ax_bar.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                        va='center', ha='left', fontsize=9, fontweight='bold')

        ax_bar.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved prediction visualization to: {output_path}")


def compute_evaluation_metrics(predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Calculates accuracy, balanced accuracy, F1, precision, recall, and confusion matrix."""
    has_ground_truth = all(p['true_label'] is not None for p in predictions)
    if not has_ground_truth:
        return None

    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, confusion_matrix
    )

    y_true = [p['true_label'] for p in predictions]
    y_pred = [p['predicted_label'] for p in predictions]

    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

    return {
        'total_samples': len(predictions),
        'accuracy': round(acc, 2),
        'balanced_accuracy': round(bal_acc, 2),
        'macro_f1': round(macro_f1, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'confusion_matrix': cm.tolist(),
        'classes': CANONICAL_CLASSES
    }


def main():
    parser = argparse.ArgumentParser(description="Monkeypox Multi-Model Inference & Evaluation Tool")
    parser.add_argument("--model", type=str, default="1", help="Model choice (1-12) or experiment folder name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth checkpoint")
    parser.add_argument("--custom-images", nargs="*", type=str, default=None, help="Paths to custom image files or directory")
    parser.add_argument("--random-test-samples", type=int, default=None, help="Number of random test images to evaluate")
    parser.add_argument("--test-dir", type=str, default="processed/test", help="Path to test dataset directory")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (TTA)")
    parser.add_argument("--output-dir", type=str, default="results/predictions", help="Directory to save visual outputs and CSVs")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MONKEYPOX DETECTION: MODEL INFERENCE & EVALUATION SYSTEM")
    print("=" * 80)

    # 1. Initialize Predictor
    predictor = MonkeypoxPredictor(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        device=args.device
    )

    # 2. Gather Images
    selected_images: List[Path] = []
    if args.custom_images:
        for p_str in args.custom_images:
            p = Path(p_str)
            if p.is_dir():
                for f in p.glob('*'):
                    if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        selected_images.append(f)
            elif p.is_file():
                selected_images.append(p)
        print(f"Loaded {len(selected_images)} custom images.")
    elif args.random_test_samples:
        test_dir = Path(args.test_dir)
        print(f"Sampling {args.random_test_samples} random images from {test_dir}...")
        selected_images = sample_random_test_images(test_dir, args.random_test_samples)
        print(f"Sampled images:\n" + "\n".join([f"  - [{p.parent.name}] {p.name}" for p in selected_images]))
    else:
        # Default fallback: sample 6 images from test set
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            print(f"No input mode specified. Defaulting to sampling 6 random images from {test_dir}...")
            selected_images = sample_random_test_images(test_dir, 6)
        else:
            print("Please specify --custom-images <paths...> or --random-test-samples <N>.")
            sys.exit(1)

    # 3. Perform Inference
    print(f"\nRunning inference (TTA={args.tta})...")
    predictions = predictor.predict_batch(selected_images, use_tta=args.tta)

    # 4. Display Results Table
    df_preds = pd.DataFrame([
        {
            'Filename': p['filename'],
            'True Label': p['true_label'] if p['true_label'] else 'N/A',
            'Predicted Label': p['predicted_label'],
            'Confidence (%)': f"{p['confidence']*100:.1f}%",
            'Chickenpox (%)': f"{p['probabilities']['Chickenpox']*100:.1f}%",
            'Monkeypox (%)': f"{p['probabilities']['Monkeypox']*100:.1f}%",
            'Normal (%)': f"{p['probabilities']['Normal']*100:.1f}%",
            'Status': ('CORRECT' if p['is_correct'] else 'INCORRECT') if p['is_correct'] is not None else 'PREDICTED'
        }
        for p in predictions
    ])

    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY TABLE")
    print("=" * 80)
    print(df_preds.to_string(index=False))

    # Save CSV
    csv_path = out_dir / "predictions_summary.csv"
    df_preds.to_csv(csv_path, index=False)
    print(f"\nSaved CSV report to: {csv_path}")

    # 5. Render Visual Cards with Annotations and Probabilities
    fig_path = out_dir / "predictions_annotated.png"
    render_prediction_visuals(predictions, fig_path, title_suffix=f"({predictor.model_identifier})")

    # 6. Compute Metrics if Ground Truth Available
    metrics = compute_evaluation_metrics(predictions)
    if metrics:
        print("\n" + "=" * 80)
        print("QUANTITATIVE METRICS (GROUND TRUTH EVALUATED)")
        print("=" * 80)
        print(f"Total Evaluated Samples: {metrics['total_samples']}")
        print(f"Accuracy:                {metrics['accuracy']:.2f}%")
        print(f"Balanced Accuracy:       {metrics['balanced_accuracy']:.2f}%")
        print(f"Macro-F1 Score:          {metrics['macro_f1']:.2f}%")
        print(f"Macro Precision:         {metrics['precision']:.2f}%")
        print(f"Macro Recall:            {metrics['recall']:.2f}%")
        print("\nConfusion Matrix (Rows: True, Cols: Pred):")
        print(f"Classes: {CANONICAL_CLASSES}")
        print(np.array(metrics['confusion_matrix']))

        with open(out_dir / "metrics_summary.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nSaved metrics JSON to: {out_dir / 'metrics_summary.json'}")

    print("\nInference and Evaluation Completed Successfully!")

if __name__ == "__main__":
    main()
