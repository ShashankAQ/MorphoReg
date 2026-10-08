import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os

import utils
from utils import ext_transforms as et
import network
from datasets import VOCSegmentation

# Suppress matplotlib GUI backend
import matplotlib
matplotlib.use('Agg')

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

def decode_prediction(pred, dataset='voc'):
    if dataset == 'voc':
        return VOCSegmentation.decode_target(pred.squeeze().cpu().numpy()).astype(np.uint8)
    raise NotImplementedError

def run_inference(image_path, model):
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

    return raw_image, mask_img, blended



st.set_page_config(page_title="Segmentation Inference App", layout="wide")




ckpt_path = ""# replace model path here
model = load_model(ckpt_path)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("Image uploaded successfully!")
    if st.button("Run model"):
        with st.spinner("Running model..."):
            original, mask, overlay = run_inference(tmp_path, model)

            st.subheader("Results")
            col1, col2, col3 = st.columns(3)

            col1.image(original, caption="Original Image", use_column_width=True)
            col2.image(mask, caption="Predicted Segmentation", use_column_width=True)
            col3.image(overlay, caption="Overlay", use_column_width=True)

        os.remove(tmp_path)

    if st.button("Reset"):
        st.experimental_rerun()
