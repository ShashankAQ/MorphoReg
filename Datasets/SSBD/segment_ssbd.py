"""Full-resolution grayscale SSBD segmentation using the PBMC DeepLabV3+ model."""
import argparse
import csv
import os
import sys
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Root folder of grayscale TIFF/PNG images; searched recursively')
    parser.add_argument('--output', required=True, type=Path, help='Output folder outside the input tree')
    parser.add_argument('--ckpt', required=True, type=Path, help='Your trusted segmentation .pth checkpoint')
    parser.add_argument('--segmentation-repo', type=Path, default=Path(__file__).resolve().parents[2] / 'segmentation', help='Folder containing network/ and utils/')
    parser.add_argument('--model', default='deeplabv3plus_resnet101')
    parser.add_argument('--num-classes', type=int, default=21)
    parser.add_argument('--output-stride', type=int, choices=[8, 16], default=16)
    parser.add_argument('--foreground-class', type=int, default=1)
    parser.add_argument('--background', choices=['median', 'keep', 'black'], default='median')
    parser.add_argument('--largest-only', action='store_true', help='Save only the largest detected object')
    parser.add_argument('--second-min-area', type=float, default=250, help='Minimum contour area for the second object, in original pixel units')
    parser.add_argument('--separable-conv', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return parser

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}

# ------------------------ Model loading ------------------------

def load_model(

    ckpt_path,

    model_name="deeplabv3plus_resnet101",

    num_classes=21,

    output_stride=16,

    separable_conv=False,

):

    model = network.modeling.__dict__[model_name](

        num_classes=num_classes,

        output_stride=output_stride,

    )

    if separable_conv and "plus" in model_name:

        network.convert_to_separable_conv(model.classifier)

    utils.set_bn_momentum(model.backbone, momentum=0.01)

    checkpoint = torch.load(

        ckpt_path,

        map_location="cpu",

        weights_only=False,

    )

    if "model_state" not in checkpoint:

        raise KeyError(

            "Checkpoint does not contain the key 'model_state'. "

            f"Available keys: {list(checkpoint.keys())}"

        )

    state = checkpoint["model_state"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state)

    device = torch.device(device_name)

    model.to(device)

    model.eval()

    print(f"Using device: {device}")

    return model, device

# ------------------------ DeepLab preprocessing ------------------------

def preprocess_for_deeplab(image_path):

    """

    Converts the source image to the format expected by the segmentation model.

    Important:

    - There is no spatial resize here.

    - The model receives the original image height and width.

    - Grayscale images are repeated into three channels.

    """

    img = Image.open(image_path)

    if img.mode == "RGBA":

        img = img.convert("RGB")

    if img.mode != "RGB":

        img_np = np.asarray(img).astype(np.float32)

        if img_np.ndim == 3:

            img_np = img_np[..., 0]

        mn = float(img_np.min())

        mx = float(img_np.max())

        if mx > mn:

            img_np = (img_np - mn) / (mx - mn)

        else:

            img_np = np.zeros_like(img_np, dtype=np.float32)

        img_8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        raw_pil = Image.fromarray(

            np.repeat(img_8[..., None], 3, axis=2),

            mode="RGB",

        )

    else:

        raw_pil = img

    dummy_label = Image.fromarray(

        np.zeros((raw_pil.height, raw_pil.width), dtype=np.uint8)

    )

    transform = et.ExtCompose(

        [

            et.ExtToTensor(),

            et.ExtNormalize(

                mean=[0.485, 0.456, 0.406],

                std=[0.229, 0.224, 0.225],

            ),

        ]

    )

    image_tensor, _ = transform(raw_pil, dummy_label)

    return image_tensor.unsqueeze(0)

# ------------------------ Image helpers ------------------------

def read_raw_gray(image_path):

    """

    Reads the original image while preserving its grayscale dtype whenever possible.

    """

    suffix = Path(image_path).suffix.lower()

    if suffix in {".tif", ".tiff"}:

        arr = tiff.imread(image_path)

    else:

        arr = np.asarray(Image.open(image_path))

    if arr.ndim != 2:

        raise ValueError(

            f"Expected a 2-D grayscale image, but received shape {arr.shape}: "

            f"{image_path}"

        )

    return arr

def save_tiff(path, arr):

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    tiff.imwrite(str(path), arr)

def replace_background_full_size(

    raw_gray,

    object_mask_u8,

    mode="median",

):

    """

    Returns a full-resolution image with no crop and no resize.

    mode:

      - "median": replace pixels outside the selected object with the median

                  intensity of the original background.

      - "keep": retain the complete original image.

      - "black": set pixels outside the selected object to zero.

    """

    if mode == "keep":

        return raw_gray.copy()

    outside = object_mask_u8 == 0

    output = raw_gray.copy()

    if mode == "black":

        output[outside] = 0

        return output

    if mode != "median":

        raise ValueError(

            f"Unsupported background_mode={mode!r}. "

            "Use 'median', 'keep', or 'black'."

        )

    if np.any(outside):

        background_value = np.asarray(

            np.median(raw_gray[outside]),

            dtype=raw_gray.dtype,

        ).item()

    else:

        background_value = np.asarray(0, dtype=raw_gray.dtype).item()

    output[outside] = background_value

    return output

def find_selected_object_masks(

    prediction,

    image_height,

    image_width,

    foreground_class=1,

    keep_second=True,

    area_threshold=250,

):

    """

    Converts the model prediction into one or two full-resolution object masks.

    Returns:

        list[tuple[str, float, np.ndarray]]

        Each tuple contains:

            object tag, contour area, full-resolution uint8 mask.

    """

    if prediction.shape != (image_height, image_width):

        prediction = cv2.resize(

            prediction,

            (image_width, image_height),

            interpolation=cv2.INTER_NEAREST,

        )

    foreground_mask = (

        prediction == foreground_class

    ).astype(np.uint8) * 255

    contours_info = cv2.findContours(

        foreground_mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE,

    )

    contours = (

        contours_info[0]

        if len(contours_info) == 2

        else contours_info[1]

    )

    contour_areas = [

        (contour, float(cv2.contourArea(contour)))

        for contour in contours

    ]

    contour_areas.sort(key=lambda item: item[1], reverse=True)

    maximum_objects = 2 if keep_second else 1

    selected = []

    for index, (contour, area) in enumerate(

        contour_areas[:maximum_objects]

    ):

        if index == 1 and area < area_threshold:

            continue

        tag = "largest" if index == 0 else "second"

        object_mask = np.zeros(

            (image_height, image_width),

            dtype=np.uint8,

        )

        cv2.drawContours(

            object_mask,

            [contour],

            contourIdx=-1,

            color=255,

            thickness=-1,

        )

        selected.append((tag, area, object_mask))

    return selected

# ------------------------ Single-image segmentation ------------------------

def segment_one_image_original_size(

    image_path,

    image_output_dir,

    mask_output_dir,

    model,

    device,

    foreground_class=1,

    keep_second=True,

    area_threshold=250,

    background_mode="median",

):

    """

    Segments one image and saves full-resolution outputs.

    No crop and no resize are applied to the saved image or mask.

    """

    raw_gray = read_raw_gray(image_path)
    input_tensor = preprocess_for_deeplab(image_path).to(device)

    with torch.inference_mode():

        logits = model(input_tensor)

        prediction = (

            logits.argmax(dim=1)

            .squeeze(0)

            .detach()

            .cpu()

            .numpy()

            .astype(np.uint8)

        )

    raw_gray = read_raw_gray(image_path)

    height, width = raw_gray.shape

    selected_masks = find_selected_object_masks(

        prediction=prediction,

        image_height=height,

        image_width=width,

        foreground_class=foreground_class,

        keep_second=keep_second,

        area_threshold=area_threshold,

    )

    if not selected_masks:

        return {

            "status": "no_object",

            "objects_saved": 0,

        }

    stem = Path(image_path).stem

    for tag, area, object_mask in selected_masks:

        segmented_full_size = replace_background_full_size(

            raw_gray=raw_gray,

            object_mask_u8=object_mask,

            mode=background_mode,

        )

        image_name = f"{stem}_{tag}.tif"

        mask_name = f"{stem}_{tag}_mask.tif"

        save_tiff(

            Path(image_output_dir) / image_name,

            segmented_full_size.astype(raw_gray.dtype, copy=False),

        )

        save_tiff(

            Path(mask_output_dir) / mask_name,

            object_mask.astype(np.uint8, copy=False),

        )

    return {

        "status": "ok",

        "objects_saved": len(selected_masks),

    }

# ------------------------ Recursive SSBD folder processing ------------------------

def list_sequence_folders(dataset_root):

    """

    Finds directories that directly contain image files.

    Example:

      Murine_PBMCs/

          PBMC-timelapse-plate4_45/

              frame files...

      Murine_immune_related_cell_lines/

          3mix-timelapse-plate16_53/

              frame files...

    """

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():

        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    sequence_folders = []

    for current_root, _, filenames in os.walk(dataset_root):

        contains_images = any(

            Path(filename).suffix.lower() in IMAGE_EXTENSIONS

            for filename in filenames

        )

        if contains_images:

            sequence_folders.append(Path(current_root))

    return sorted(sequence_folders)

def process_ssbd_dataset_original_size(

    dataset_root,

    output_root,

    model,

    device,

    foreground_class=1,

    keep_second=True,

    area_threshold=250,

    background_mode="median",

):

    """

    Recursively processes an SSBD dataset while preserving its sequence-folder

    organization and every source image's original spatial dimensions.

    Output structure:

      output_root/

          <sequence_name>/

              images_segmented_original_size/

              masks_original_size/

    """

    dataset_root = Path(dataset_root)

    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    sequence_folders = list_sequence_folders(dataset_root)

    if not sequence_folders:

        raise RuntimeError(

            f"No TIFF/PNG image folders were found under: {dataset_root}"

        )

    records = []
    total_images = 0

    successful_images = 0

    no_object_images = 0

    failed_images = 0

    total_objects = 0

    print(f"\nDataset root: {dataset_root}")

    print(f"Sequence folders found: {len(sequence_folders)}")

    for sequence_dir in tqdm(

        sequence_folders,

        desc=f"Sequences: {dataset_root.name}",

    ):

        relative_sequence = sequence_dir.relative_to(dataset_root)

        sequence_output = output_root / relative_sequence

        image_output_dir = (

            sequence_output / "images_segmented_original_size"

        )

        mask_output_dir = (

            sequence_output / "masks_original_size"

        )

        image_output_dir.mkdir(parents=True, exist_ok=True)

        mask_output_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(

            path

            for path in sequence_dir.iterdir()

            if path.is_file()

            and path.suffix.lower() in IMAGE_EXTENSIONS

        )

        for image_path in image_files:

            total_images += 1

            try:

                result = segment_one_image_original_size(

                    image_path=image_path,

                    image_output_dir=image_output_dir,

                    mask_output_dir=mask_output_dir,

                    model=model,

                    device=device,

                    foreground_class=foreground_class,

                    keep_second=keep_second,

                    area_threshold=area_threshold,

                    background_mode=background_mode,

                )

                records.append({"image": str(image_path.relative_to(dataset_root)), **result, "error": ""})
                if result["status"] == "ok":

                    successful_images += 1

                    total_objects += result["objects_saved"]

                else:

                    no_object_images += 1

                    print(f"No object found: {image_path}")

            except Exception as exc:

                records.append({"image": str(image_path.relative_to(dataset_root)), "status": "failed", "objects_saved": 0, "error": str(exc)})
                failed_images += 1

                print(f"Failed: {image_path}\nReason: {exc}")

    print("\nProcessing summary")

    print("------------------")

    print(f"Dataset:              {dataset_root.name}")

    print(f"Sequence folders:     {len(sequence_folders)}")

    print(f"Images inspected:     {total_images}")

    print(f"Images segmented:     {successful_images}")

    print(f"No-object images:     {no_object_images}")

    print(f"Failed images:        {failed_images}")

    print(f"Object files saved:   {total_objects}")

    print(f"Output root:          {output_root}")

    with (output_root / 'processing_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'status', 'objects_saved', 'error'])
        writer.writeheader()
        writer.writerows(records)
    return failed_images

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if not args.input.is_dir():
        parser.error('--input must be an existing folder')
    if args.input == args.output or args.input in args.output.parents or args.output in args.input.parents:
        parser.error('Input and output folders must be separate, non-overlapping trees')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('--output must be a new or empty folder to avoid overwriting results')
    if not args.ckpt.is_file():
        parser.error('--ckpt must point to an existing checkpoint file')
    if args.num_classes < 2 or not 0 <= args.foreground_class < args.num_classes:
        parser.error('Check --num-classes and --foreground-class')
    if args.second_min_area < 0:
        parser.error('--second-min-area must be nonnegative')
    repo = args.segmentation_repo.resolve()
    if not all((repo / name / '__init__.py').is_file() for name in ['network', 'utils']):
        parser.error(f'{repo} must contain network/ and utils/. Supply --segmentation-repo.')
    sys.path.insert(0, str(repo))
    global cv2, np, Image, tqdm, torch, tiff, utils, et, network
    try:
        import cv2
        import numpy as np
        from PIL import Image
        from tqdm import tqdm
        import torch
        import tifffile as tiff
        import utils
        from utils import ext_transforms as et
        import network
    except ImportError as exc:
        parser.error(f'Missing dependency: {exc}. Install segmentation/requirements.txt and opencv-python tifffile.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if args.model not in network.modeling.__dict__ or not args.model.startswith('deeplab'):
        parser.error('Unknown DeepLab model name')
    model, device = load_model(args.ckpt, args.model, args.num_classes,
                               args.output_stride, args.separable_conv, device_name)
    failures = process_ssbd_dataset_original_size(
        args.input, args.output, model, device,
        foreground_class=args.foreground_class,
        keep_second=not args.largest_only,
        area_threshold=args.second_min_area,
        background_mode=args.background,
    )
    return 1 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())
