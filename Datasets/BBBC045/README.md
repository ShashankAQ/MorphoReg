# BBBC045 — Human White Blood Cells

## Dataset link

[BBBC045 · Version 1 — Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC045)

Download the TIFF image montages using the
**BBBC045_Stained_Montages.zip** link on the dataset page.
Compensated image files are also available as
**BBBC045_Stained_Populations.zip**.

## Dataset description

BBBC045 contains images of human white blood cells from 13 healthy
donors, acquired using an Amnis ImageStream100 imaging flow cytometer.

The cell counts recorded in our dataset documentation are:

| Cell type | Number of cells |
|---|---:|
| Neutrophils | 62,379 |
| T cells | 25,210 |
| Monocytes | 4,346 |
| B cells | 4,128 |
| Eosinophils | 4,042 |
| **Total** | **100,105** |

Blood was collected into heparinized tubes and divided into
500 µL aliquots. Individual aliquots were stained with FITC-labelled
antibodies against CD3 (T cells), CD14 (monocytes), CD15 (neutrophils),
or CD19 (B cells).

The released data include compensated image files (`.cif`) and TIFF
image montages (`.tif`), containing fluorescence, brightfield, and
darkfield channels. Each montage contains 900 individual images.
Some images contain partial cells.

Cell-type ground truth was determined using fluorescence-based
gating. Labels are represented by the cell-type directory names
within each donor's folder.

## Use in MorphoReg

BBBC045 is used as an external, unseen dataset to evaluate transfer
from the model trained on the private ETH dataset.

The processing workflow includes extracting individual grayscale
brightfield cell images from the montages and applying the
DeepLabV3+ segmentation pipeline.

[BBBC045.ipynb](BBBC045.ipynb) contains the montage extraction
workflow. It processes channel-5 TIFF montages and extracts
55 × 55 pixel cell images from 1650 × 1650 pixel montages.
Its Google Colab/Drive paths should be updated for your environment.

The dataset itself is downloaded separately and is not included
in this repository.

## References

1. Nassar, M., et al. (2019).
   **Label-Free Identification of White Blood Cells Using Machine Learning.**
   *Cytometry Part A*, 95(8), 836–842.
   https://doi.org/10.1002/cyto.a.23794

2. Ljosa, V., Sokolnicki, K. L., and Carpenter, A. E. (2012).
   **Annotated high-throughput microscopy image sets for validation.**
   *Nature Methods*, 9, 637.
   https://doi.org/10.1038/nmeth.2083

Dataset accession: **BBBC045v1**, Broad Bioimage Benchmark Collection.