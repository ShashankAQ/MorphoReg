# ImageFlowCytometryTcellData

## Dataset and paper links

- [Dataset v1: Image flow cytometry data of T-cells from healthy and Sezary patients](https://zenodo.org/records/5391155)
- [Dataset DOI](https://doi.org/10.5281/zenodo.5391155)
- [Paper DOI](https://doi.org/10.1016/j.crmeth.2021.100094)
- [Open-access paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9017143/)

## References

Otesteanu, C. F., et al. (2021). **A weakly supervised deep learning approach for
label-free imaging flow-cytometry-based blood diagnostics.**
*Cell Reports Methods*, 1(6), 100094.
https://doi.org/10.1016/j.crmeth.2021.100094

Otesteanu, C. F., Ugrinic, M., Holzner, G., Chang, Y.-T., Fassnacht, C.,
Guenova, E., Stavrakis, S., deMello, A., and Claassen, M. (2021).
**Image flow cytometry data of T-cells from healthy and Sezary patients** (v1).
Zenodo. https://doi.org/10.5281/zenodo.5391155

## Task and folder layout

The source study evaluates patient-level **healthy donor (HD) versus Sézary
syndrome (SS)** classification using label-free brightfield imaging of sorted
T cells, with four healthy donors and five SS patients. A patient diagnosis is
not a ground-truth label saying that every individual cell is pathological.
The scripts here perform segmentation and preprocessing, not diagnosis or
reproduction of the iCellCnn classifier.

Expected input: `ImageFlowCytometryTcellData/<donor>/<numeric subfolder>/<image>`.
For example: `HD1/1/image.tif`. Single-frame 2-D grayscale TIFF/PNG files are
supported. RGB images and stacks are rejected. Direct image files in each
numeric folder are processed; other subfolder layouts are not searched.

By default, all HD*/SS* directories are discovered, including HD3 if present.
The explicit folder list below reproduces your selection of eight donor folders;
it omits HD3. Missing requested donor folders are reported and skipped.

## Install

Run from the MorphoReg root, with the PBMC DeepLab code in `segmentation/`:

```powershell
python -m pip install -r Datasets/ImageFlowCytometryTcellData/requirements.txt
```

The scripts import `network` and `utils` from that folder automatically. An
alternative clone can be supplied using `--segmentation-repo`.

## Segment

```powershell
python Datasets/ImageFlowCytometryTcellData/segment_ifc.py `
  --input "C:\Users\shash\Downloads\ImageFlowCytometryTcellData\ImageFlowCytometryTcellData" `
  --output "C:\ETH\ETH_BROAD_IFC" `
  --ckpt "C:\ETH\models\SOTA.pth" `
  --folders HD1 HD2 HD4 SS1 SS2 SS3 SS4 SS5
```

Output must be new or empty and outside the input tree. Use a trusted checkpoint
containing `model_state`. Defaults are `deeplabv3plus_resnet101`, 21 classes,
output stride 16 and foreground class 1; these must match your checkpoint.
CUDA is selected automatically; `--device cpu` forces CPU execution.

Your original `build_zoomed_in_280_tif` definition was not supplied. This is a new,
explicit implementation of the requested parameters, not a byte-equivalent
reproduction of that function. Model input is min-max normalized to uint8,
ImageNet-normalized and bilinearly resized to 513 x 513. Use
`--deeplab-crop-size 0` for native-size inference. Predicted masks are mapped back
to native resolution with nearest-neighbor interpolation. Filled external
contours are used; interior mask holes are filled.

The largest object is retained, plus a second if its native contour area is at
least 250 pixels squared. Use `--largest-only` or `--second-min-area` to change it.
Native segmented images retain original dimensions, dtype and foreground values.
Background defaults to median grey (`--background median`); `black` and `keep`
are also available. It also saves 160 x 160 isotropically zoomed images and masks,
centered on the object's bounding box. Defaults: `--out-size 160`,
`--target-cell-frac 0.70`, `--min-upscale 1.1`, `--max-upscale 6.0`.
Scale is reduced when necessary to fit the whole bounding box within the canvas.
Zoomed images use bilinear interpolation, which changes texture and pixel values;
use native images when morphology analysis requires native sampling.

Output folders: `native_images/`, `native_masks/`, `zoomed_images/`,
`zoomed_masks/`, each preserving donor/subfolder organization.
`segmentation_manifest.csv` records the source image, output paths, scale,
no-object cases and failures. A nonzero exit code signals processing failures.

## Audit and exclude debris

The audit reuses the supplied ETH script's border-background median/MAD threshold,
binary closing, hole filling, largest-component selection and halo dilation.
Measurements are computed on native segmented images, not zoomed outputs.
They are heuristic audit measurements, not ground-truth cell masks.

```powershell
python Datasets/ImageFlowCytometryTcellData/filter_debris.py `
  --manifest "C:\ETH\ETH_BROAD_IFC\segmentation_manifest.csv" `
  --review-csv "C:\ETH\IFC_review.csv"
```

The CSV flags area fraction >= 0.20, aspect ratio >= 2.0 in the ETH elongated
candidate range, and bounding-box extent >= 100 in the ETH large-extent range.
Thresholds can be changed with `--area-threshold`, `--aspect-threshold`,
`--extent-threshold`, and `--halo-dilate`. The ETH random samples previously
reviewed are not reused: they do not identify reviewed cells in this dataset.
No flags automatically establish debris, and no files are removed during audit.
Large or irregular SS cells may be biologically relevant; inspect the images
rather than excluding them solely for their morphology. These ETH thresholds
have not been validated for this dataset.

Open the CSV, inspect the flagged images, and enter **keep** or **debris** in the
`decision` column for every row with status `ok`. Then run:

```powershell
python Datasets/ImageFlowCytometryTcellData/filter_debris.py `
  --review-csv "C:\ETH\IFC_review.csv" `
  --apply `
  --output "C:\ETH\IFC_filtered"
```

Reviewed objects and their paired native/zoomed masks are copied into `keep/`
and `debris/`. Each object has a numbered folder to avoid filename collisions.
Use `keep/` for downstream work. Original files remain intact. The output must be
new or empty. `filter_manifest.csv` records the copy destinations. Audit failures
must be investigated separately; they are not included in the filtered output.
Keep every cell from a donor in the same train/validation/test partition.

## Validation

Python syntax and CLI help were checked. Synthetic checks covered heuristic
blank/object detection, uint16 foreground preservation, reviewed keep/debris
copying with paired masks, retention of originals and overwrite protection.
DeepLab inference, image interpolation and dataset performance have not been
validated here because PyTorch/OpenCV, the checkpoint and source images are not
available in this environment.
