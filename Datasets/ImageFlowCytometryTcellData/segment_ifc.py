"""Segment HD/SS donor folders and save native and zoomed grayscale TIFFs."""
import argparse
import csv
import sys
from pathlib import Path

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

def zoom_cell(raw, mask, out_size=160, target_cell_frac=0.70,
              min_upscale=1.1, max_upscale=6.0, background_mode='median'):
    """Center a masked cell on a square canvas; isotropic interpolation is explicit."""
    ys, xs = np.where(mask != 0)
    if not len(xs):
        raise ValueError('Cannot zoom an empty mask')
    extent = max(int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
    scale = float(np.clip(out_size * target_cell_frac / extent, min_upscale, max_upscale))
    # Preserve the bounding box when the requested minimum scale would clip it.
    scale = min(scale, (out_size-4) / extent)
    cx = (float(xs.min()) + float(xs.max())) / 2
    cy = (float(ys.min()) + float(ys.max())) / 2
    shift = (out_size-1) / 2
    matrix = np.array([[scale, 0, shift-scale*cx], [0, scale, shift-scale*cy]], dtype=np.float32)
    native = replace_background_full_size(raw, mask, background_mode)
    outside = mask == 0
    bg = float(np.median(raw[outside])) if np.any(outside) and background_mode != 'black' else 0
    zoomed = cv2.warpAffine(native, matrix, (out_size, out_size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    zoom_mask = cv2.warpAffine(mask, matrix, (out_size, out_size), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return native, zoomed, zoom_mask, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ckpt', required=True, type=Path)
    parser.add_argument('--folders', nargs='+', help='Donors to process; default discovers all HD*/SS* directories')
    parser.add_argument('--segmentation-repo', type=Path, default=Path(__file__).resolve().parents[2]/'segmentation')
    parser.add_argument('--model', default='deeplabv3plus_resnet101')
    parser.add_argument('--num-classes', type=int, default=21)
    parser.add_argument('--output-stride', type=int, choices=[8,16], default=16)
    parser.add_argument('--foreground-class', type=int, default=1)
    parser.add_argument('--deeplab-crop-size', type=int, default=513, help='Model-input square resize, not a saved-image crop; 0 uses native input')
    parser.add_argument('--out-size', type=int, default=160)
    parser.add_argument('--target-cell-frac', type=float, default=.70)
    parser.add_argument('--min-upscale', type=float, default=1.1)
    parser.add_argument('--max-upscale', type=float, default=6.0)
    parser.add_argument('--largest-only', action='store_true')
    parser.add_argument('--second-min-area', type=float, default=250)
    parser.add_argument('--background', choices=['median','black','keep'], default='median')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args()
    root, out = args.input.resolve(), args.output.resolve()
    if not root.is_dir() or not args.ckpt.is_file():
        parser.error('Input folder and checkpoint must exist')
    if root == out or root in out.parents or out in root.parents:
        parser.error('Input and output must be separate non-overlapping trees')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Output must be new or empty')
    if not (0 < args.target_cell_frac < 1 and 0 < args.min_upscale <= args.max_upscale):
        parser.error('Invalid zoom settings')
    if args.out_size < 8 or args.deeplab_crop_size < 0 or args.second_min_area < 0:
        parser.error('Invalid image size or area threshold')
    if not 0 <= args.foreground_class < args.num_classes:
        parser.error('Foreground class must be in the model class range')
    repo = args.segmentation_repo.resolve()
    if not all((repo/n/'__init__.py').is_file() for n in ['network','utils']):
        parser.error('Segmentation repository must contain network/ and utils/')
    donors = args.folders or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(('HD','SS')))
    if not donors or any(Path(d).name != d or d in ('.','..') for d in donors):
        parser.error('No donor folders found, or invalid donor names')
    files = []
    for donor in donors:
        donor_dir = root/donor
        if not donor_dir.is_dir():
            print(f'Skipping missing donor: {donor}')
            continue
        for sub in sorted((p for p in donor_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p:int(p.name)):
            files.extend(sorted(p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in {'.tif','.tiff','.png'}))
    if not files:
        parser.error('No images found in donor/numeric-subfolder layout')
    global np, cv2, torch, tiff, Image, utils, et, network
    sys.path.insert(0,str(repo))
    import numpy as np
    import cv2
    import torch
    import tifffile as tiff
    from PIL import Image
    import utils
    from utils import ext_transforms as et
    import network
    from tqdm import tqdm
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model, device = load_model(args.ckpt, args.model, args.num_classes, args.output_stride, device_name=device_name)
    out.mkdir(parents=True,exist_ok=True)
    fields=['source','status','native_image','native_mask','zoomed_image','zoomed_mask','contour_area','scale','error']
    failures=0
    with (out/'segmentation_manifest.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for path in tqdm(files,desc='HD/SS segmentation'):
            try:
                raw=read_raw_gray(path)
                tensor=preprocess_for_deeplab(path).to(device)
                if args.deeplab_crop_size:
                    tensor=torch.nn.functional.interpolate(tensor,size=(args.deeplab_crop_size,)*2,mode='bilinear',align_corners=False)
                with torch.inference_mode():
                    prediction=model(tensor).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                objects=find_selected_object_masks(prediction,*raw.shape,args.foreground_class,not args.largest_only,args.second_min_area)
                if not objects:
                    writer.writerow({'source':str(path),'status':'no_object'})
                for tag,area,mask in objects:
                    native,zoomed,zoom_mask,scale=zoom_cell(raw,mask,args.out_size,args.target_cell_frac,args.min_upscale,args.max_upscale,args.background)
                    # Include original extension to prevent foo.png/foo.tif collisions.
                    relative=path.relative_to(root).parent/(path.name+'_'+tag+'.tif')
                    paths={key:out/folder/relative for key,folder in [('native_image','native_images'),('native_mask','native_masks'),('zoomed_image','zoomed_images'),('zoomed_mask','zoomed_masks')]}
                    for key,array in [('native_image',native),('native_mask',mask),('zoomed_image',zoomed),('zoomed_mask',zoom_mask)]:
                        save_tiff(paths[key],array)
                    writer.writerow({'source':str(path),'status':'ok','contour_area':area,'scale':scale,**{k:str(v) for k,v in paths.items()}})
                handle.flush()
            except Exception as exc:
                failures+=1
                writer.writerow({'source':str(path),'status':'failed','error':str(exc)})
                handle.flush()
                print(f'Failed {path}: {exc}')
    print(f'Manifest: {out / "segmentation_manifest.csv"}; failures: {failures}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
