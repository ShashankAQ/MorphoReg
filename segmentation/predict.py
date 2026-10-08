# predict.py

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import utils
from utils import ext_transforms as et
import network
from datasets import VOCSegmentation
import argparse
from tqdm import tqdm

# ------------------------ Model Loading ------------------------
def load_model(ckpt_path, model_name='deeplabv3plus_resnet101', num_classes=21, output_stride=16, separable_conv=False):
    model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
    if separable_conv and 'plus' in model_name:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = torch.nn.DataParallel(model)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model

# ------------------------ Preprocessing ------------------------
def preprocess(image_path, crop_size=513):
    raw_image = Image.open(image_path).convert('RGB')
    dummy_label = Image.fromarray(np.zeros((raw_image.height, raw_image.width), dtype=np.uint8))
    transform = et.ExtCompose([
        et.ExtResize(crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(raw_image, dummy_label)
    return image_tensor.unsqueeze(0), raw_image

# ------------------------ Decode Mask ------------------------
def decode_prediction(pred, dataset='voc'):
    if dataset == 'voc':
        return VOCSegmentation.decode_target(pred.squeeze().cpu().numpy()).astype(np.uint8)
    raise NotImplementedError

# ------------------------ Inference on Single Image ------------------------
def run_single_inference(image_path, ckpt_path, output_path='output.png'):
    model = load_model(ckpt_path)
    image_tensor, raw_image = preprocess(image_path)
    image_tensor = image_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(1)

    decoded = decode_prediction(pred)
    mask_img = Image.fromarray(decoded).resize(raw_image.size)
    raw_array = np.array(raw_image)
    mask_array = np.array(mask_img)

    overlay = np.zeros_like(raw_array)
    mask_binary = np.any(mask_array != [0, 0, 0], axis=-1)
    overlay[mask_binary] = [255, 0, 0]

    blended = (0.6 * raw_array + 0.4 * overlay).astype(np.uint8)

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(raw_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_array)
    plt.title("Predicted Segmentation")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# ------------------------ Inference on Folder ------------------------
def run_batch_inference_with_filtering(input_dir, output_dir, ckpt_path):
    model = load_model(ckpt_path)
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]

    for img_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, img_file)
        image_tensor, raw_image = preprocess(image_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(1).squeeze().cpu().numpy()

        pred_resized = cv2.resize(pred.astype(np.uint8), raw_image.size[::-1], interpolation=cv2.INTER_NEAREST)
        mask = (pred_resized == 1).astype(np.uint8) * 255
        raw_gray = np.array(raw_image.convert("L"))

        contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        filtered_mask = np.zeros_like(mask)

        for cnt in contours:
            blob_mask = np.zeros_like(mask)
            cv2.drawContours(blob_mask, [cnt], -1, color=255, thickness=-1)
            blob_pixels = raw_gray[blob_mask > 0]
            mean_intensity = np.mean(blob_pixels)
            std_intensity = np.std(blob_pixels)

            if mean_intensity >= 1 and std_intensity >= 1 and cv2.contourArea(cnt) > 0:
                cv2.drawContours(filtered_mask, [cnt], -1, color=255, thickness=-1)

        overlay = np.zeros((*filtered_mask.shape, 3), dtype=np.uint8)
        overlay[filtered_mask > 0] = [255, 0, 0]
        raw_array = np.array(raw_image)
        blended = (0.6 * raw_array + 0.4 * overlay).astype(np.uint8)
        output_image = Image.fromarray(blended)

        save_path = os.path.join(output_dir, img_file)
        output_image.save(save_path)

    print(f"\n✅ All filtered overlays saved to: {output_dir}")

# ------------------------ Main Entry ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepLabv3+ Inference on PBMC images")
    parser.add_argument('--mode', choices=['single', 'batch'], required=True, help='Run mode: "single" for one image, "batch" for folder')
    parser.add_argument('--image_path', type=str, help='Path to input image (required for single mode)')
    parser.add_argument('--input_dir', type=str, help='Input folder path (required for batch mode)')
    parser.add_argument('--output_dir', type=str, help='Output folder path (required for batch mode)')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_image', type=str, default='output.png', help='Path to save output image (for single mode)')

    args = parser.parse_args()

    if args.mode == 'single':
        if not args.image_path:
            raise ValueError("Please provide --image_path for single mode")
        run_single_inference(args.image_path, args.ckpt, args.output_image)
    elif args.mode == 'batch':
        if not args.input_dir or not args.output_dir:
            raise ValueError("Please provide both --input_dir and --output_dir for batch mode")
        run_batch_inference_with_filtering(args.input_dir, args.output_dir, args.ckpt)
