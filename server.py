import logging.config
# from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer
import transformers
from transformers.pipelines import PIPELINE_REGISTRY
import threading
import time
from queue import Queue
import streamlit as st
from annotated_text import annotated_text
from collections import Counter, OrderedDict
import pandas as pd
import altair as alt
import io

# @param ["yolov6s", "yolov6n", "yolov6t"]
checkpoint: str = "runs\\train\\self_attention\\weights\\best_ckpt"
# checkpoint:str ="yolov6s6" #@param ["yolov6s", "yolov6n", "yolov6t"]
device: str = "cpu"  # @param ["gpu", "cpu"]
half: bool = False  # @param {type:"boolean"}


import os
import requests
import torch
import math
import cv2
import numpy as np
import PIL
# Change directory so that imports wortk correctly
if os.getcwd() == "/content":
    os.chdir("YOLOv6")
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer

from typing import List, Optional
# Download weights
if not os.path.exists(f"{checkpoint}.pt"):
    print("Downloading checkpoint...")
    os.system(
        f"""wget -c https://github.com/meituan/YOLOv6/releases/download/0.3.0/{checkpoint}.pt""")

# Set-up hardware options
cuda = device != 'cpu' and torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')


def preProcess_image(img):
    """function to enhance image and add extra edge detect layer"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create a CLAHE object.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    # apply CLAHE to the image
    claheImg = clahe.apply(img)

    # de-noise and blur CLAHE image
    deNoise = cv2.fastNlMeansDenoising(claheImg, None, 20, 7, 30)
    blur = cv2.GaussianBlur(deNoise, (5, 5), 5)

    # apply canny edge detect
    edgeDetectImg = cv2.Canny(blur, 45, 55)

    # combine the images into a 2 channel image
    claheImg = cv2.cvtColor(claheImg, cv2.COLOR_GRAY2BGR)
    edgeDetectImg = cv2.cvtColor(edgeDetectImg, cv2.COLOR_GRAY2BGR)
    # Convert
    claheImg = claheImg.transpose(
        (2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    edgeDetectImg = edgeDetectImg.transpose(
        (2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    channels = [claheImg, edgeDetectImg]
    newImg = np.concatenate(channels, axis=0)
    return newImg


def check_img_size(img_size, s=32, floor=0):
    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(
            f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size, list) else [new_size] * 2


def process_image(image_data, img_size, stride, half):
    '''Process image before image inference.'''
    try:
        from PIL import Image
        img_src = np.asarray(Image.open(io.BytesIO(image_data)))
        assert img_src is not None, f'Invalid image: {image_data}'
    except Exception as e:
        LOGGER.Warning(e)

    image = letterbox(img_src, img_size, stride=stride)[0]
    image = preProcess_image(image)

    # Convert
    # image = image.transpose((2, 0, 1))  # HWC to CHW
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src


model = DetectBackend(f"./{checkpoint}.pt", device=device)
stride = model.stride
class_names = load_yaml("./data/dataset.yaml")['names']

if half & (device.type != 'cpu'):
    model.model.half()
else:
    model.model.float()
    half = False

if device.type != 'cpu':
    model(torch.zeros(
        1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup


def predict_on_image(img_data):
    hide_labels: bool = False  # @param {type:"boolean"}
    hide_conf: bool = False  # @param {type:"boolean"}

    img_size: int = 640  # @param {type:"integer"}

    conf_thres: float = .25  # @param {type:"number"}
    iou_thres: float = .45  # @param {type:"number"}
    max_det: int = 1000  # @param {type:"integer"}
    agnostic_nms: bool = False  # @param {type:"boolean"}

    img_size = check_img_size(img_size, s=stride)

    img, img_src = process_image(img_data, img_size, stride, half)

    img = img.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)[0]
    classes: Optional[List[int]] = None  # the classes to keep
    det = non_max_suppression(
        pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    img_ori = img_src.copy()
    if len(det):
        det[:, :4] = Inferer.rescale(
            img.shape[2:], det[:, :4], img_src.shape).round()
        for *xyxy, conf, cls in reversed(det):
            class_num = int(cls)
            label = None if hide_labels else (
                class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
            Inferer.plot_box_and_label(img_ori, max(round(sum(
                img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
    out_img = PIL.Image.fromarray(img_ori)
    return out_img


st.title("Dental Caries Detector")

uploaded_files = st.file_uploader(
    "Choose an image:", type=["jpg", "png", "jpeg"], accept_multiple_files=True)


if uploaded_files is not None:
    output_images = []
    for index, file in enumerate(uploaded_files):
        out_image = predict_on_image(file.read())
        # output_images.append(out_image)
        st.image(out_image, caption=f"{file.name} prediction")
