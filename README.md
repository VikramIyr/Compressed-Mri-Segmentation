# Brain Tumor Segmentation from Compressed MRI Slices

This repository contains the code used for the project  
**“Brain Tumor Segmentation from Compressed MRI Slices: Quantifying the Compression–Accuracy Trade-off with 2D U-Net”**,  
developed for the **ETH Zürich – Deep Learning (2025)** course.

The goal of this project is to study how **explicit MRI input compression**—via slice skipping and spatial downsampling—affects segmentation quality, using a fixed 2D U-Net architecture on the BraTS 2021 dataset.

---

## 1. Project Overview

We investigate two compression strategies:

- **Slice skipping**: retaining only every *k*-th axial slice  
- **Spatial downsampling**: reducing in-plane resolution by factor *r*

We evaluate the resulting trade-offs using:
- Dice score 
- HD95 
- Tumor volume error 

All experiments:
- use a **standard 2D U-Net**
- are trained **from scratch**

---

## 2. Repository Structure

```text
.
├── data/                     
│   └── BraTS2021_XXXXX/
│
├── src/
│   ├── train.py              
│   ├── eval.py               
│   ├── datasets.py           
utilities
│
├── scripts/
│   └── make_figures.py       
 figures
│
├── runs/
│   ├── baseline/            
│   ├── stride2/              
│   ├── down2/                
│   ├── s2_d2/                
│   └── eval/
│       ├── metrics.csv       
│       └── summary.csv       
results
│
├── figures/                  
│
├── requirements.txt
└── README.md
```

## 3. Environment Setup

### 3.1 Python Environment
Python 3.8+ is recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```



### 3.2 Hardware
Experiments were run on:

- Single NVIDIA GPU (GTX 1660 Ti, 6GB VRAM)

CPU-only evaluation is supported, but training will be very slow.

## 4. Dataset Setup (BraTS 2021)
Download BraTS 2021 from the official source.

Extract into the data/ directory:

```text
data/
└── BraTS2021_00000/
└── BraTS2021_00001/
└── ...
```
Each case directory must contain:

- *_flair.nii.gz
- *_t1ce.nii.gz (optional)
- *_seg.nii.gz

### Dataset Sanity Check


Run:

```bash
python src/sanity_check_dataset.py
```
## 5. Training Experiments
All training runs use the same architecture and optimizer, differing only in compression settings.

### 5.1 Baseline (No Compression)

```bash
python src/train.py \
  --run_dir runs/baseline \
  --epochs 10 \
  --batch_size 8 \
  --modalities flair \
  --slice_stride 1 \
  --downsample 1 \
  --loss bce
```

### 5.2 Slice Skipping (k = 2)

```bash
python src/train.py \
  --run_dir runs/stride2 \
  --epochs 10 \
  --batch_size 8 \
  --modalities flair \
  --slice_stride 2 \
  --downsample 1
  --loss bce
```
### 5.3 Spatial Downsampling (r = 3)

```bash
python src/train.py \
  --run_dir runs/down2 \
  --epochs 10 \
  --batch_size 8 \
  --modalities flair \
  --slice_stride 3 \
  --downsample 1
  --loss bce
```
### 5.4 Spatial Downsampling (r = 2)

```bash
python src/train.py \
  --run_dir runs/down2 \
  --epochs 10 \
  --batch_size 8 \
  --modalities flair \
  --slice_stride 1 \
  --downsample 2
  --loss bce
```


### 5.5 Combined Compression (k = 2, r = 2)

```bash
python src/train.py \
  --run_dir runs/s2_d2 \
  --epochs 10 \
  --batch_size 8 \
  --modalities flair \
  --slice_stride 2 \
  --downsample 2
  --loss bce
```

### 5.6 Baseline Dice Loss (No Compression)

```bash
python src/train.py \
  --run_dir runs/baseline_dice \
  --epochs 10 \
  --batch_size 8 \
  --modalities flair \
  --slice_stride 1 \
  --downsample 1 \
  --loss dice
```
### 5.7 Slice Skipping Dice Loss(k = 2)
```bash
python src/train.py \
  --run_dir runs/stride2_dice \
  --epochs 10 \
  --batch_size 8 \
  --modalities flair \
  --slice_stride 2 \
  --downsample 1 \
  --loss dice
```
Each run automatically saves:

```text
runs/<run_name>/best.pt
```
based on validation Dice score.

## 6. Evaluation

### 6.1 Run Evaluation for the models

```bash
python src/eval.py \
  --ckpt runs/baseline/best.pt \
  --modalities flair \
  --slice_stride 1 \
  --downsample 1 \
  --split val
```
```bash
python src/eval.py \
  --ckpt runs/stride2/best.pt \
  --modalities flair \
  --split val \
  --slice_stride 2 \
  --downsample 1 \
  ```
```bash
python src/eval.py \
  --ckpt runs/down2/best.pt \
  --modalities flair \
  --split val \
  --slice_stride 1 \
  --downsample 2 \
  ```
```bash
python src/eval.py \
  --ckpt runs/s2_d2/best.pt \
  --modalities flair \
  --split val \
  --slice_stride 2 \
  --downsample 2 \
```
CSV file:

```text
runs/eval/metrics.csv
```

### 6.2 Aggregate Multiple Experiments
Each evaluation appends results to:

```text
runs/eval/summary.csv
```
You can find the model weights here: https://drive.google.com/drive/folders/1VulB8fbILH1lGaE6aeTNv4yDAlR9i0vb?usp=sharing

To be put under the folder runs in a folder of the name of the type.

## 7. Figure Generation 

### Qualitative Compression Figure

To generate the qualitative comparison figure (same subject and slice under different compression settings):

```bash
python scripts/figure_qualitative_compression.py \
  --cpu \
  --data_root data \
  --case_id BraTS2021_00000 \
  --slice_idx 80 \
  --modalities flair \
  --ckpt_baseline runs/baseline/best.pt \
  --ckpt_stride2 runs/stride2/best.pt \
  --ckpt_down2 runs/down2/best.pt \
  --ckpt_s2d2 runs/s2_d2/best.pt
```



