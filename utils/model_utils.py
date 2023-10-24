import numpy as np
import torch
from PIL.Image import Image
from typing import Union
from torchvision.transforms import v2
from torchvision.models.detection import KeypointRCNN

from models.yolo import Model
from yolo_utils.general import non_max_suppression_face
from utils.data_utils import post_process_yolo_preds


def generate_face_bbox(
    model: Model, image: Union[Image, str], transforms: v2.Compose
) -> np.ndarray:
    input_img = image.resize((640, 640))
    input_tens = transforms(input_img)
    device = model.device
    input_tens = input_tens.to(device)
    if len(input_tens.shape) == 3:
        input_tens = input_tens.unsqueeze(0)
    output = model(input_tens)
    preds = post_process_yolo_preds(output, image)
    return preds


def generate_keypoints(
    model: KeypointRCNN,
    image: Union[Image, str],
    transforms: v2.Compose,
    detect_threshold=0.9,
) -> torch.Tensor:
    input_tens = transforms(image)
    device = model.device
    input_tens = input_tens.to(device)
    if len(input_tens.shape) == 3:
        input_tens = input_tens.unsqueeze(0)
    output = model(input_tens)

    # Filter out Keypoints for Objects w/ low prob of being human
    scores = output[0]["scores"]
    kpts = output[0]["keypoints"]
    idx = torch.where(scores > detect_threshold)
    keypoints = kpts[idx]

    # Keep only upper body keypoints
    keypoints = keypoints[:, :11, :]
    return keypoints
