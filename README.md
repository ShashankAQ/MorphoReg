# MorphoReg
### Morphology-Aware Register Self-Supervision for Brightfield Cell Microscopy

<p align="center">
  <b>Native-scale microscopy SSL with morphology-guided masking, register bottlenecks, and donor-aware evaluation.</b>
</p>

<p align="center">
  <!-- Replace these once public -->
  <a href="#paper">Paper</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#method-at-a-glance">Method</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#repository-structure">Code</a>
</p>

---

> **TL;DR.** MorphoReg is a self-supervised vision transformer for single-cell brightfield microscopy.  
> Instead of resizing every cell to a common apparent size, MorphoReg preserves native morphology, allocates tokens through multi-resolution native canvases, applies cell-aware masked image modeling, and routes information through a small set of learnable morphology registers.

---

## Why MorphoReg?

Brightfield cell images are visually simple at first glance, but the discriminative signal is often subtle: cell size, contour, halo structure, local texture, nucleus-like internal contrast, and the spatial relationship between the cell and its immediate context.

Standard natural-image SSL pipelines can unintentionally destroy or dilute those cues by aggressively resizing crops, applying scale-changing augmentations, or treating every patch as equally informative.

MorphoReg is built around a different principle:

> **Preserve the physical morphology first; learn invariance second.**

The framework is designed for microscopy settings where apparent cell size and local morphology are part of the signal rather than nuisance variation.

---

## Method at a glance

<p align="center">
  <img src="assets/morphoreg_overview.png" width="95%" alt="MorphoReg overview">
</p>

> **Figure placeholder.** Replace `assets/morphoreg_overview.png` with the final paper overview figure.

MorphoReg combines five ideas:

1. **Native-scale views**  
   Cells are not globally resized to a fixed apparent size. Each cell is placed into the smallest valid canvas from **80 / 96 / 128 / 160 px**, preserving native morphology while keeping computation efficient.

2. **Mask-guided spatial regions**  
   Supplied segmentation masks define cell core, periphery, near-cell context, and far background. These regions are used to guide token masking and spatial conditioning.

3. **Cell-aware masked image modeling**  
   Masked blocks are sampled preferentially from biologically informative regions instead of uniformly over the image.

4. **Morphology register bottleneck**  
   After an initial transformer stem, token information is compressed through a small set of learnable registers using a  
   **tokens → registers → registers → tokens** interaction pattern.

5. **Multi-view self-distillation**  
   Two native-scale views are paired with a high-detail branch. The objective combines global self-distillation, masked patch prediction, register consistency, detail-to-native consistency, and register diversity.

---

## What is different from standard ViT/iBOT pretraining?

| Component | Standard image SSL | MorphoReg |
|---|---|---|
| Cell scale | Often altered by crop/resize | **Preserved** |
| View size | Usually fixed | **80 / 96 / 128 / 160 native buckets** |
| Masking | Spatially uniform | **Cell-aware / morphology-aware** |
| Context | Generic image context | **Core / periphery / near / far regions** |
| Registers | Optional global tokens | **Explicit morphology bottleneck** |
| Detail branch | Usually absent | **Dedicated high-detail view** |
| Microscopy masks | Often unused | **Used to guide regions + masking** |

---

## Dataset preparation

Our current ETH brightfield corpus contains:

- **91,357** cleaned image-mask pairs
- all source images at **160 × 160**
- **90,096 uint16** images used for the first MorphoReg pretraining run
- **1,261 uint8** images excluded from the primary pretraining run to avoid mixing intensity encodings
- supplied masks paired one-to-one with the cleaned segmentations

The model never uses filename terms such as `stained`, `unstained`, or `unknown` as supervision or sampling labels.

### Native-canvas distribution

For the 90,096 uint16 training cells:

| Canvas | Cells |
|---:|---:|
| 80 × 80 | 86,023 |
| 96 × 96 | 1,348 |
| 128 × 128 | 784 |
| 160 × 160 | 1,941 |

This keeps most cells compact while retaining native cell size.

---

## Architecture

The default model is a compact ViT-S-style backbone:

- patch size: **8**
- embedding dimension: **384**
- depth: **12**
- attention heads: **6**
- stem depth: **2**
- morphology registers: **4**
- drop path: **0.1**
- CLS projection dimension: **8192**
- patch projection dimension: **8192**

After the two standard transformer stem blocks, the remaining blocks use the morphology-register bottleneck.

### SSL objective

The total objective is:

```text
L = λ_cls L_cls
  + λ_patch L_patch
  + λ_reg L_reg
  + λ_detail L_detail
  + λ_div L_diversity
```

Default weights:

```text
λ_cls       = 1.00
λ_patch     = 1.00
λ_reg       = 0.25
λ_detail    = 0.10
λ_diversity = 0.02
```

---

## Repository structure

```text
MorphoReg/
│
├── segmentation/
│   ├── README.md
│   └── ...
│
├── ssl/
│   ├── train_morphoreg.py
│   ├── morphoreg_data.py
│   ├── morphoreg_model.py
│   ├── morphoreg_loss.py
│   ├── compute_morphoreg_stats.py
│   └── ...
│
├── unsupervised/
│   ├── README.md
│   └── ...
│
├── evaluation/
│   ├── bbbc045/
│   └── ...
│
├── assets/
│   └── morphoreg_overview.png
│
└── README.md
```

The top-level repository is intentionally lightweight. Each subdirectory contains its own task-specific README and reproducible commands.

---

## Quick Start

### 1. Prepare paired brightfield images and masks

Expected layout:

```text
data/
├── segmentations/
│   ├── cell_0001.tif
│   ├── cell_0002.tif
│   └── ...
└── masks/
    ├── cell_0001_mask.tif
    ├── cell_0002_mask.tif
    └── ...
```

Build the audit manifest:

```bash
python build_morphoreg_audit.py \
  --image_dir /path/to/segmentations \
  --mask_dir /path/to/masks \
  --output /path/to/eth_morphoreg_audit.csv
```

### 2. Compute native-view normalization statistics

```bash
python compute_morphoreg_stats.py \
  --audit_csv /path/to/eth_morphoreg_audit.csv \
  --output /path/to/eth_morphoreg_stats_uint16.json \
  --dtype uint16 \
  --margin 12
```

### 3. Train MorphoReg

```bash
python train_morphoreg.py \
  --audit_csv /path/to/eth_morphoreg_audit.csv \
  --stats_json /path/to/eth_morphoreg_stats_uint16.json \
  --output_dir /path/to/run \
  --epochs 150 \
  --batch_size 32 \
  --patch_size 8 \
  --embed_dim 384 \
  --depth 12 \
  --stem_depth 2 \
  --num_heads 6 \
  --num_registers 4 \
  --mim_ratio 0.30 \
  --lr 5e-4 \
  --warmup_epochs 10 \
  --lambda_cls 1.0 \
  --lambda_patch 1.0 \
  --lambda_reg 0.25 \
  --lambda_detail_reg 0.10 \
  --lambda_diversity 0.02
```

The trainer uses:

- mixed precision by default
- EMA teacher updates
- cosine LR decay
- increasing weight decay
- gradient clipping
- checkpointing after every epoch

### Resume training

```bash
python train_morphoreg.py \
  --audit_csv /path/to/eth_morphoreg_audit.csv \
  --stats_json /path/to/eth_morphoreg_stats_uint16.json \
  --output_dir /path/to/run \
  --epochs 150 \
  --batch_size 32 \
  --resume /path/to/run/checkpoint.pth
```

---

## Evaluation

We evaluate representations on downstream brightfield white-blood-cell classification using **BBBC045**.

A central goal of the evaluation is to compare learned representations against classical morphology features under the **same cells, same donor splits, and same classifier protocol**.

### Exact paired CellProfiler vs learned-feature comparison

For the current BBBC045 CellProfiler comparison:

- released cells: **100,105**
- CellProfiler complete-case cohort: **77,121**
- exact iBOT alignment: **77,121 / 77,121**
- CellProfiler BF features: **128**
- learned CLS representation: **1,536 dimensions**
- donor-level LOSO evaluation
- donor-wise random undersampling performed before LOSO for the repository-matched protocol

We report:

- B vs T lymphocyte classification
- 4-class classification:
  - Lymphocyte
  - Eosinophil
  - Monocyte
  - Neutrophil
- per-donor results
- paired representation differences
- repeated donor-wise undersampling

> Final MorphoReg downstream results will be added here after the training/evaluation run is complete.

---

## Reproducibility

We aim to make every important choice explicit:

- exact paired image-mask manifest
- fixed native-view normalization statistics
- deterministic seed handling
- explicit dtype filtering
- bucket-aware batching
- checkpointable training
- exact donor-held-out evaluation
- paired CellProfiler / learned-feature comparisons on identical cells

For each experiment, the output directory stores the configuration and checkpoints required to reproduce the run.

---

## Current training configuration

The primary MorphoReg v1 run uses:

```text
dataset            ETH brightfield
training images    90,096 uint16 cells
patch size         8
native buckets     80 / 96 / 128 / 160
detail branch      160 × 160
backbone            ViT-S-style
embed dim          384
depth              12
stem depth         2
registers          4
MIM ratio          0.30
epochs             150
batch size         32
optimizer          AdamW
teacher            EMA
```

---

## Design philosophy

MorphoReg is not intended to treat microscopy as a smaller version of ImageNet.

The design is built around three microscopy-specific assumptions:

**1. Scale can be signal.**  
Cell size and contour should not be normalized away by default.

**2. Not all patches are equally informative.**  
The cell interior, boundary, and immediate context deserve more attention than distant background.

**3. A small latent morphology bottleneck can encourage structured representations.**  
Registers are used as an explicit communication bottleneck rather than passive extra tokens.

---

## Paper

**MorphoReg: Morphology-Aware Register Self-Supervision for Brightfield Cell Microscopy**

Authors: *[Author names]*  
Conference: *CVPR [Year]*  
Paper: *[arXiv / OpenAccess link]*  
Project page: *[URL]*

---

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{morphoreg2027,
  title     = {MorphoReg: Morphology-Aware Register Self-Supervision for Brightfield Cell Microscopy},
  author    = {Author One and Author Two and ...},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {20XX}
}
```

---

## License

Code: *[choose license, e.g. MIT / Apache-2.0]*

Dataset redistribution may be subject to the licenses and terms of the original data providers. Please consult the corresponding dataset documentation before redistribution.

---

## Acknowledgements

We thank the creators of the ETH brightfield microscopy data, BBBC045, CellProfiler, and the open-source self-supervised learning community whose tools and datasets made this work possible.

---

<p align="center">
  <b>Morphology should be preserved, not resized away.</b>
</p>
