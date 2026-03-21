# LayerCAM: Explainable AI for Chest X-Ray Pathology Detection

A pure PyTorch + NumPy implementation of **LayerCAM** applied to multi-label chest pathology classification on the [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert/data) dataset. Generates visual heatmap explanations for ResNet-50 predictions, with GradCAM as a comparison baseline.

**Authors:** Kirill Shumskiy, Ivan Makarov, Arthur Babkin

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Generating Explanations](#generating-explanations)
- [Architecture](#architecture)
- [Experiment Results](#experiment-results)
  - [Training](#training-run)
  - [Classification Performance](#classification-performance-validation-set)
  - [GradCAM vs LayerCAM: Layer Comparison](#gradcam-vs-layercam-layer-comparison)
  - [CAM Analysis: TP / FP / FN / TN](#cam-analysis-tp--fp--fn--tn)

---

## Overview

Deep learning models for medical imaging are black boxes — clinicians need to know *why* a model predicted a pathology, not just *that* it did. This project implements **LayerCAM**, a class activation mapping method that produces fine-grained spatial explanations by weighting feature map activations element-wise with their gradients.

Key properties of this implementation:
- **No external XAI libraries** — LayerCAM and GradCAM are implemented from scratch in PyTorch/NumPy
- **Multi-label classification** — each X-ray can have multiple simultaneous diagnoses
- **Uncertainty-aware training** — handles CheXpert's `-1` (uncertain) labels via masked losses
- **Multi-layer CAM fusion** — can aggregate explanations across multiple ResNet layers

---

## Method

### LayerCAM

LayerCAM generates class activation maps using element-wise gradient weighting:

```
LayerCAM(i, j, c) = Σ_k  A^k_{ij} · ReLU( ∂y^c / ∂A^k_{ij} )
```

where `y^c` is the logit for class `c` and `A^k_{ij}` are activations at position `(i, j)` in layer `k`.

**Difference from GradCAM:**

| Method | Weighting strategy |
|--------|---|
| GradCAM | Global mean of gradients per channel → scalar weight per channel |
| LayerCAM | Pixel-wise gradient × activation → spatial weight at every location |

LayerCAM preserves spatial detail lost by GradCAM's global pooling, producing sharper, more localized heatmaps.

### Multi-layer fusion

`generate_layer_specific_cam()` generates CAMs from several ResNet layers, resizes them to a common resolution, and averages them — combining coarse semantic features (deep layers) with fine spatial detail (shallow layers).

---

## Dataset

**CheXpert** — 224,316 frontal and lateral chest radiographs from 65,240 patients (Stanford).

Download: [Kaggle — CheXpert](https://www.kaggle.com/datasets/ashery/chexpert/data)

### 13 target pathology labels

- Enlarged Cardiomediastinum
- Cardiomegaly
- Lung Opacity
- Lung Lesion
- Edema
- Consolidation
- Pneumonia
- Atelectasis
- Pneumothorax
- Pleural Effusion
- Pleural Other
- Fracture
- Support Devices

### Label encoding

| Value | Meaning |
|-------|---------|
| `1` | Present |
| `0` | Absent |
| `-1` | Uncertain |
| `NaN` | Not mentioned |

Uncertain (`-1`) and missing labels are excluded from loss computation via a **valid mask** — only clearly labeled samples contribute gradients for each class.

---

## Project Structure

```
s26_xai_layercam/
├── docs/
│   ├── proposal/
│   │   ├── proposal.pdf          # Project proposal
│   │   └── proposal.tex          # LaTeX source
│   └── photos/
│       ├── resnet50.jpg                          # ResNet-50 architecture diagram
│       ├── layergradcam.png                      # LayerCAM pipeline diagram
│       ├── roc_auc_curves.png                    # Per-label ROC curves (validation set)
│       ├── explanation_Lung_Opacity.png          # GradCAM vs LayerCAM (layer3/4) — Lung Opacity
│       ├── explanation_Pleural_Effusion.png      # GradCAM vs LayerCAM (layer3/4) — Pleural Effusion
│       ├── explanation_Support_Devices.png       # GradCAM vs LayerCAM (layer3/4) — Support Devices
│       ├── cam_analysis_Lung_Opacity.png         # TP/FP/FN/TN analysis — Lung Opacity
│       ├── cam_analysis_Pleural_Effusion.png     # TP/FP/FN/TN analysis — Pleural Effusion
│       ├── cam_analysis_Support_Devices.png      # TP/FP/FN/TN analysis — Support Devices
│       └── cam_analysis_*.png                    # TP/FP/FN/TN analysis — remaining classes
├── src/
│   ├── dataset.py                # CheXpert PyTorch Dataset
│   ├── preprocess.py             # CSV parsing and label cleaning
│   ├── model.py                  # ResNet-50 for multi-label classification
│   ├── loss.py                   # MaskedBCELoss, MaskedFocalLoss
│   ├── train.py                  # Training pipeline with TensorBoard logging
│   ├── evaluate.py               # Per-label AUC, threshold selection, ROC curves
│   ├── gradcam.py                # GradCAM (baseline XAI method)
│   ├── layercam.py               # LayerCAM (main XAI method)
│   └── experiment.ipynb          # End-to-end CheXpert experiment notebook
├── checkpoints/
│   └── best_model.pth            # Best validation checkpoint
├── requirements.txt
└── README.md
```

### Module descriptions

| File | Responsibility |
|------|---------------|
| `dataset.py` | `CheXpertDataset` — loads images from disk, returns `(tensor, targets, valid_mask)` |
| `preprocess.py` | Filters to frontal views, fixes paths, handles "No Finding" label logic |
| `model.py` | `CheXpertResNet50` — pretrained ResNet-50 with custom head; exposes `get_feature_maps()` for CAM hooks |
| `loss.py` | `MaskedBCELoss` and `MaskedFocalLoss` — compute loss only on valid labels |
| `train.py` | `CheXpertTrainer` — training loop, LR scheduling, per-class AUC/F1, checkpoint saving |
| `evaluate.py` | `collect_predictions()`, `compute_per_label_auc()`, `find_optimal_thresholds()` — validation-set evaluation and per-label threshold selection |
| `gradcam.py` | `GradCAM` class + `visualize_cam()`, `evaluate_cam_quality()` |
| `layercam.py` | `LayerCAM` class + `visualize_layercam()`, `compare_gradcam_layercam()`, `analyze_cam_differences()` |

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd s26_xai_layercam

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA optional (CPU and Apple Silicon MPS supported)

---

## Usage

### Training

```bash
python src/train.py \
  --data_root /path/to/chexpert \
  --batch_size 32 \
  --num_epochs 50 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --loss_type bce \
  --log_dir logs \
  --checkpoint_dir checkpoints
```

Use `--loss_type focal` with `--alpha 1.0 --gamma 2.0` for focal loss (recommended for class-imbalanced data).

The trainer expects the standard CheXpert directory layout:
```
/path/to/chexpert/
├── train.csv
├── valid.csv
└── CheXpert-v1.0-small/
    └── train/
        └── patient*/
```

**Outputs:**
- `checkpoints/best_model.pth` — saved when validation AUC improves
- `logs/` — TensorBoard event files

Monitor training with:
```bash
tensorboard --logdir logs
```

#### Training hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 32 | Batch size |
| `--num_epochs` | 50 | Training epochs |
| `--learning_rate` | 1e-4 | Adam learning rate |
| `--weight_decay` | 1e-5 | Adam weight decay |
| `--loss_type` | `bce` | `bce` or `focal` |
| `--alpha` | 1.0 | Focal loss alpha |
| `--gamma` | 2.0 | Focal loss gamma |

The learning rate is halved (factor=0.5) if validation loss does not improve for 3 epochs.

---

### Generating Explanations

```python
import torch
from src.model import CheXpertResNet50
from src.layercam import LayerCAM, visualize_layercam, compare_gradcam_layercam
from src.gradcam import GradCAM

TARGET_CLASSES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# Load trained model
model = CheXpertResNet50(num_classes=13)
ckpt = torch.load("checkpoints/best_model.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Generate LayerCAM for all classes
layercam = LayerCAM(model, target_layer_name="backbone.layer4")
cams = layercam.generate_multi_class_cam(input_tensor)  # shape: (13, H, W)

# Visualize overlay on original image
visualize_layercam(original_image, cams, class_names=TARGET_CLASSES, save_path="explanation.png")

# Side-by-side GradCAM vs LayerCAM comparison
gradcam = GradCAM(model, target_layer_name="backbone.layer4")
compare_gradcam_layercam(original_image, gradcam, layercam, input_tensor,
                         class_idx=9, class_name="Pleural Effusion")

# Quantitative comparison
from src.layercam import analyze_cam_differences
metrics = analyze_cam_differences(gradcam_map, layercam_map)
# Returns: correlation, mean/max/std difference, focus_ratio
```

#### Multi-layer fusion

```python
# Average CAMs from multiple ResNet layers
cam_fused, per_layer_cams = layercam.generate_layer_specific_cam(
    input_tensor,
    target_class_idx=4,
    layer_names=["backbone.layer2", "backbone.layer3", "backbone.layer4"]
)
```

---

## Architecture

```
[CheXpert CSV]
      │
      ▼
[preprocess.py]  ── frontal-only filter, path fixing, label cleaning
      │
      ▼
[dataset.py]     ── CheXpertDataset → (image_tensor, targets, valid_mask)
      │
      ▼
[train.py]       ── CheXpertTrainer
      ├── [model.py]  CheXpertResNet50 (ImageNet pretrained → 13-class head)
      └── [loss.py]   MaskedBCELoss / MaskedFocalLoss
      │
      ▼
[Saved checkpoint]
      │
      ├── [gradcam.py]   GradCAM  ── global channel weights
      └── [layercam.py]  LayerCAM ── element-wise spatial weights
                │
                ▼
         [Heatmap overlays / comparison plots]
```

**Model:** ResNet-50 (ImageNet pretrained, 23.5M parameters) with final FC replaced by `Linear(2048, 13)`. The `get_feature_maps()` method exposes `layer4` activations for CAM hook registration.

**Images:** Resized to 224×224, normalized with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

---

## Experiment Results

The full experiment is documented in [`src/experiment.ipynb`](src/experiment.ipynb).

### Training run

| Setting | Value |
|---------|-------|
| Train / val split | 10,000 / 2,000 images |
| Model | ResNet-50 (23.5M parameters) |
| Device | Apple Silicon MPS |
| Loss | BCE with valid mask |
| Optimizer | Adam, lr=1e-4, wd=1e-5 |
| Epochs | 10 |

**Epoch-level summary:**

| Epoch | Train Loss | Val Loss | Train AUC | Val AUC | Train F1 | Val F1 |
|------:|-----------|---------|----------|--------|---------|-------|
| 1 | — | — | — | — | — | — |
| 8 (best) | 0.0135 | 0.6495 | 0.9999 | **0.8145** | 0.9964 | 0.6774 |

**Overfitting observation:** Training AUC reaches 0.9999 by epoch 8, while validation AUC plateaus around 0.81. The model clearly memorises the training split;

---

### Classification performance (validation set)

Per-label ROC-AUC, optimal decision threshold (Youden's J), and F1 at that threshold:

| Label | AUC | Threshold | F1 |
|---|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.7245 | 0.2128 | 0.5382 |
| Cardiomegaly | 0.9115 | 0.4259 | 0.8477 |
| Lung Opacity | 0.8672 | 0.7812 | 0.8735 |
| Lung Lesion | 0.8098 | 0.2759 | 0.6180 |
| Edema | 0.8919 | 0.7337 | 0.8495 |
| Consolidation | — | — | — |
| Pneumonia | 0.8373 | 0.1488 | 0.5857 |
| Atelectasis | 0.8442 | 0.6470 | 0.8163 |
| Pneumothorax | 0.7732 | 0.2248 | 0.5719 |
| Pleural Effusion | 0.9222 | 0.7878 | 0.8985 |
| Pleural Other | 0.7826 | 0.1399 | 0.4167 |
| Fracture | 0.7515 | 0.2110 | 0.5674 |
| Support Devices | 0.8110 | 0.6889 | 0.8986 |
| **Mean** | **0.8307** | | **0.7068** |

Thresholds vary widely (0.14–0.79) due to class imbalance — rare classes (Consolidation, Pneumonia, Pleural Other) produce low sigmoid outputs and require low thresholds. A single global threshold of 0.5 would severely underdetect these classes. Per-label thresholds are selected by maximising Youden's J (sensitivity + specificity − 1) on the validation set and stored alongside the model checkpoint.

**Best-performing classes:** Pleural Effusion (0.9222), Cardiomegaly (0.9115), Edema (0.8919)
**Weakest-performing classes:** Enlarged Cardiomediastinum (0.7245), Fracture (0.7515), Pneumothorax (0.7732)

### ROC curves

![ROC curves per label](docs/photos/roc_auc_curves.png)

---

### GradCAM vs LayerCAM: Layer Comparison

We compared both methods across two ResNet layers — `backbone.layer3` (shallower, finer resolution) and `backbone.layer4` (deeper, coarser) — on the three most frequently positive classes in our validation subset: **Support Devices**, **Pleural Effusion**, and **Lung Opacity**.

**Quantitative agreement (Pearson r between heatmaps):**

| Class | layer4 r | layer3 r | Interpretation |
|-------|:--------:|:--------:|----------------|
| Support Devices | 0.79 | 0.43 | Diverge at shallow layers |
| Pleural Effusion | 1.00 | 0.89 | Near-identical across both layers |
| Lung Opacity | 0.98 | 0.54 | Diverge at shallow layers |

**Per-class observations:**

- **Support Devices** — At `layer4` both methods broadly agree (r = 0.79). At `layer3` they diverge (r = 0.43): LayerCAM focuses on the upper mediastinum where devices are located, while GradCAM remains diffuse across the lungs.
- **Pleural Effusion** — Near-perfect agreement at both layers (r = 1.00 / 0.89); difference maps are blank. Both correctly localise to the right costophrenic angle. For spatially compact, high-contrast pathologies the two methods are interchangeable.
- **Lung Opacity** — Agreement at `layer4` (r = 0.98), divergence at `layer3` (r = 0.54): LayerCAM highlights specific dense foci, GradCAM activates the lungs uniformly.

**Explanation overlays (GradCAM left · LayerCAM right, layer3 top · layer4 bottom):**

![Explanation — Support Devices](docs/photos/explanation_Support_Devices.png)
*Support Devices: layer3 divergence (r = 0.43) clearly visible — LayerCAM narrows to upper mediastinum.*

![Explanation — Pleural Effusion](docs/photos/explanation_Pleural_Effusion.png)
*Pleural Effusion: both maps are nearly identical at both layers (r = 1.00 / 0.89).*

![Explanation — Lung Opacity](docs/photos/explanation_Lung_Opacity.png)
*Lung Opacity: layer3 divergence (r = 0.54) — LayerCAM highlights focal dense areas, GradCAM activates broadly.*

**Key takeaway:** The critical difference is spatial precision at shallow layers. GradCAM pools gradients globally per channel, which dilutes fine-grained spatial information for diffuse pathologies. LayerCAM's element-wise weighting preserves this detail, producing sharper, more anatomically specific heatmaps — especially at `layer3` and for distributed findings like Lung Opacity and Support Devices. Both methods are representative and reliable for prominent, localised pathologies (Pleural Effusion, AUC 0.92).

> **Note.** As students without clinical expertise in radiology, we could not independently assess whether highlighted regions correspond to anatomically plausible disease locations. For the anatomical judgements above we used SOTA LLM assistance (Claude Opus 4.6, extended thinking). The layer-level analysis design, ROC-AUC evaluation, per-label threshold selection, and all implementation were done by us — students.

---

### CAM Analysis: TP / FP / FN / TN

For each of the three key classes we sampled up to two examples per prediction category (True Positive, False Positive, False Negative, True Negative) using `backbone.layer4` and per-class optimal thresholds.

![CAM analysis — Support Devices](docs/photos/cam_analysis_Support_Devices.png)
![CAM analysis — Pleural Effusion](docs/photos/cam_analysis_Pleural_Effusion.png)
![CAM analysis — Lung Opacity](docs/photos/cam_analysis_Lung_Opacity.png)

**TP** — In all three classes, both maps attend to roughly the expected regions. For Pleural Effusion this is the lower hemithorax; for Lung Opacity the lung fields; for Support Devices a broad mediastinal area. The latter is important: for devices there is no clear focus on a specific tube or catheter, only on overall context. This suggests that the spatial resolution of `layer4` is insufficient for thin linear objects, so the model is learning the type of image rather than the device itself. LayerCAM is consistently slightly more compact; GradCAM is broader.

**FN** — The most illustrative case is Support Devices (`score = 0.096`). At a very low logit, GradCAM becomes less informative: global averaging over gradients suppresses weak local support for the class and amplifies background noise, so the map becomes nearly empty or shifts toward image borders. LayerCAM preserves residual local regions that partially support the class, but their contribution was not strong enough to affect the final decision — hence the warm area in the lower chest despite the low score. For Pleural Effusion the FN case lies very close to the decision threshold; both maps still focus on the lower lung fields. This is not a localisation miss, but a threshold miss.

**FP** — In FP cases for Lung Opacity and Pleural Effusion, both methods converge on the same intrathoracic region with high confidence. When GradCAM and LayerCAM agree in an FP case, this more likely indicates label noise or ambiguous annotation than a random hallucination. The Support Devices FP (`score = 0.952`) focuses on the lower mediastinum, likely reflecting a shortcut: the model learned a co-occurrence pattern in which an image of a severely ill patient implies support devices.

**TN** — Here, CAM maps are least reliable. GradCAM often shifts toward image corners and borders, indirectly suggesting the model relies partly on border effects and artifacts. LayerCAM remains within the thoracic cavity, but that does not mean it is correct — only that it is more anatomically plausible. In TN examples, neither method has anything specific to show because the absence of pathology has no single spatial localisation.

**Conclusion** — LayerCAM is genuinely more useful for visual auditing: it more often stays within anatomically meaningful regions and better preserves local class-supporting evidence. GradCAM is coarser and suffers more from global averaging, so in low-score and TN examples it drifts toward borders, background, and artifacts more often. This does not necessarily make it worse — such maps can reveal shortcut dependencies and the model's sensitivity to non-anatomical context. Agreement between the two methods on the same intrathoracic region is better interpreted not as a direct sign of correctness, but as an indicator of a stable pattern that either deserves greater trust or careful inspection for label noise and co-occurrence bias.

> **Note.** As students without formal clinical expertise in radiology, we were not able to independently determine whether the highlighted regions corresponded to anatomically plausible disease locations. The conclusions above were partially informed by assistance from SOTA LLMs (Claude Opus 4.6 with extended thinking and OpenAI ChatGPT GPT-5 Thinking). We also performed our own visual inspection and independently identified several patterns — notably the markedly different behaviour of the two methods for low-probability labels. The role of the LLMs was to validate and refine these observations; final conclusions were made by us as students.

---

## References

- **LayerCAM:** Jiang et al., *LayerCAM: Exploring Hierarchical Class Activation Maps for Localization* (2021)
- **GradCAM:** Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (2017)
- **CheXpert:** Irvin et al., *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison* (2019)
- **ResNet:** He et al., *Deep Residual Learning for Image Recognition* (2016)
