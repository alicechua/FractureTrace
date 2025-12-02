# FractureTrace - Fusing DINOv2 and TextCAM for Fracture Classification and Detection

This repository contains the code for our CSC2503 project **“Fusing DINOv2 and TextCAM for Fracture Classification and Detection: A Comparative Vision Transformer Study.”** It implements a weakly supervised pipeline for **localizing bone fractures in radiographs** using CNN and Vision Transformer (ViT) backbones, with **TextCAM** providing text-grounded explanations.

---

## 1. Project Overview

Most clinical-grade fracture detection systems rely on expert-drawn bounding boxes or segmentation masks, which are expensive and rarely public. In this project we:

- Benchmark a range of **CAM-style methods** for fracture **classification + localization**.
- Study **gradient/activation-based explanations** on both CNNs and ViTs.
- Propose a **DINOv2 + register tokens + TokenGrad + TextCAM** pipeline that:
  - Works with **image-level labels only** (no fracture masks).
  - Produces **spatially coherent heatmaps** and **radiology-like phrases** that describe fracture patterns.{index=1}  

---

## 2. Methods Implemented

The repo includes code/experiments for the following families of models and explainers:

### CNN baselines
- **ResNet-18 / ResNet-50**
  - Trained for multi-class fracture classification.
  - Explanations via **Grad-CAM** and **Score-CAM**.

### Vision Transformers
- **ViTmiX**
  - Custom ViT with a **patch-mixing augmentation** for regularisation.
- **DINOv2 (base)**
  - Used as a pretrained backbone with different fine-tuning strategies (frozen linear head vs. partial fine-tuning).
- **ViT + LeGrad**
  - Explainer based on **LeGrad** applied under different training regimes: no fine-tune, linear probe, and full supervised fine-tuning. 
- **OpenCLIP ViT + LeGrad**
  - Zero-shot classification with fracture prompts.
  - Linear probe on frozen OpenCLIP image embeddings to study how text supervision changes LeGrad maps.{index=5}  
- **ViT + TokenGrad**
  - A Grad-CAM–style method adapted to ViT patch tokens, aggregating gradients and activations over blocks to produce patch-level heatmaps. 

### ViT with Self-Distilled Registers
- **Register-augmented ViT** (student) distilled from a standard ViT (teacher).
- Adds **K learnable register tokens per block**; only registers + last block + final norm are trained with a patch-token distillation loss (MSE + cosine).
- TokenGrad is then applied to the student to study how registers change interpretability.

### TextCAM (Text-grounded explanations)

- **ResNet-50 + GradCAM + TextCAM**
- **DINOv2 + registers + TokenGrad + TextCAM**

TextCAM maps channel/patch importance scores into the **CLIP image–text embedding space**, then retrieves top-k phrases from a fracture vocabulary to describe “what” each highlighted region represents.

Our **proposed solution** is:

> **DINOv2 with register tokens + TokenGrad + TextCAM**,  
> which achieved the best trade-off between accuracy, fracture localization quality, and semantic alignment across all models. 

---

## 3. Dataset

We use the [Bone Break Classification dataset](https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset/data) from Kaggle, which contains:

- ~**1129 X-ray images**.
- **10 fracture classes**: Avulsion, Comminuted, Dislocation, Greenstick, Hairline, Impacted, Longitudinal, Oblique, Pathological, Spiral.
- Original split: 989 train / 140 test images.
- We create a **stratified 80/20 validation split** on the training set, yielding roughly **796 / 193 / 140** images for train / val / test.

Images are resized to **224×224**, preserving aspect ratio via scaling + zero-padding.

---

## 4. Repository Structure

This project is organised as a small set of experiment notebooks plus a shared descriptor file:

| File | Description |
|------|-------------|
| `ResNet_GradCAM.ipynb` | ResNet-18/50 baseline training on the Bone Break dataset and generation of Grad-CAM / Score-CAM heatmaps for fracture localisation. |
| `ViT_LeGrad.ipynb` | ViT training (different fine-tuning regimes) and LeGrad-based explanations on ViT backbones; also used as the base for OpenCLIP + LeGrad comparisons. |
| `ViT_TokenGrad.ipynb` | Implementation and evaluation of TokenGrad for ViT models, producing gradient–activation heatmaps over patch tokens. |
| `ViT_Self-Distilled_Registers.ipynb` | Teacher–student experiments for ViT with self-distilled register tokens (post-hoc register distillation) and TokenGrad visualisations of the resulting student. |
| `TextCAM_ResNet50.ipynb` | TextCAM on top of ResNet-50 + Grad-CAM: generates text-grounded explanations and figures for the CNN baseline. |
| `TextCAM_dinov2_with_registers.ipynb` | TextCAM on DINOv2 with register tokens + TokenGrad: main notebook for our proposed DINOv2-with-registers + TextCAM pipeline. |
| `fracture_xray_descriptors.json` | Vocabulary of fracture-related phrases (e.g., “transverse line”, “wedge fragment”, “near growth plate”) used by TextCAM to map channel importance into natural-language descriptors. |

---

## 5. Getting Started (Notebooks)

Each notebook is self-contained and can be run in any order to reproduce the corresponding figures and tables from the report.
