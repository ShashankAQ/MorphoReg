# SSBD — 237-Jin-CellDynamics

## Dataset links

- [Dataset repository and downloads](https://ssbd.riken.jp/repository/ssbd-repos-000237/)
- [Image collection](https://ssbd.riken.jp/database/237-Jin-CellDynamics/)
- [Dataset DOI](https://doi.org/10.24631/ssbd.repos.2022.06.237)

Accession: **ssbd-repos-000237**

## Reference

Jin, J., Ogawa, T., Hojo, N., Kryukov, K., Shimizu, K., Ikawa, T.,
Imanishi, T., Okazaki, T., and Shiroguchi, K. (2023).
**Robotic data acquisition with deep learning enables cell image-based
prediction of transcriptomic phenotypes.**
*Proceedings of the National Academy of Sciences*, 120(1), e2210283120.

https://doi.org/10.1073/pnas.2210283120

## Segmentation and preprocessing

`segment_ssbd.py` applies the PBMC DeepLabV3+ segmentation model to
SSBD grayscale brightfield images.

The script:

- Recursively processes single-frame grayscale TIFF and PNG images.
- Preserves original image dimensions without cropping or resizing the saved images.
- Preserves original pixel values and dtype inside the selected object mask.
- Replaces the background with median grey by default.
- Saves segmented TIFF images and separate binary masks.
- Preserves the input folder hierarchy.
- Records processing outcomes in `processing_summary.csv`.

The model input is min-max scaled to 8-bit and ImageNet-normalized.
This scaling is used for inference; saved foreground pixels come
from the original image.

Objects are selected independently in each frame. The script does
not track cell identities across time.

## Required files

Keep the script at `Datasets/SSBD/segment_ssbd.py`.

The script automatically loads the local `network` and `utils`
packages from MorphoReg's `segmentation/` folder.

Download the SSBD data separately and supply your trained `.pth`
segmentation checkpoint.

## Install dependencies

Run these commands from the MorphoReg root folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r segmentation/requirements.txt
python -m pip install opencv-python tifffile
```

## Run segmentation

Replace the example paths with your own:

```powershell
python Datasets/SSBD/segment_ssbd.py `
  --input "C:\data\SSBD_brightfield_images" `
  --output "C:\data\SSBD_segmented" `
  --ckpt "C:\ETH\models\SOTA.pth"
```

The input folder should contain only the grayscale images intended
for segmentation. RGB images and TIFF stacks are rejected.

The output folder must be new or empty and must not overlap the
input folder.

Only load a checkpoint you trust. The checkpoint must contain
`model_state`.

### Model configuration

Defaults match the supplied segmentation configuration:

| Parameter | Default |
|---|---|
| Model | `deeplabv3plus_resnet101` |
| Number of classes | `21` |
| Output stride | `16` |
| Foreground class | `1` |
| Background | `median` |
| Minimum second-object contour area | `250` |

The architecture and class count must match your checkpoint.

To set the model parameters explicitly:

```powershell
python Datasets/SSBD/segment_ssbd.py `
  --input "C:\data\SSBD_brightfield_images" `
  --output "C:\data\SSBD_segmented" `
  --ckpt "C:\ETH\models\SOTA.pth" `
  --model deeplabv3plus_resnet101 `
  --num-classes 21 `
  --output-stride 16 `
  --foreground-class 1
```

### Optional settings

- `--largest-only`: save only the largest detected object.
- `--second-min-area 250`: set the second-object minimum contour area.
- `--background median`: replace background with its median intensity.
- `--background black`: replace background with zero.
- `--background keep`: retain the original background.
- `--device cpu`: force CPU inference.
- `--device cuda`: require CUDA.
- `--segmentation-repo "C:\path\to\clone"`: use another folder containing
  the DeepLab `network` and `utils` packages.

Show all available arguments:

```powershell
python Datasets/SSBD/segment_ssbd.py --help
```

## Outputs

Each sequence folder produces:

- `images_segmented_original_size/`
- `masks_original_size/`

The largest object is saved for each frame containing a detected object.
A second object is also saved when its contour area meets the threshold,
unless `--largest-only` is supplied.

Masks are filled external contours, so interior holes are filled.
Inspect the masks against the original images before downstream analysis.

## Validation status

Python syntax, command-line argument handling, and preservation of
16-bit foreground values have been checked. Model inference and
segmentation quality still require validation with the SSBD images
and the trained checkpoint.