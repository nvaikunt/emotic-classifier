import numpy as np
import torch
from PIL.Image import Image
from typing import Union
from torchvision.transforms import v2
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.ssd import SSD

from models.yolo import Model
from yolo_utils.general import non_max_suppression_face
from utils.data_utils import post_process_yolo_preds
import math


def generate_face_bbox(
    model: Model, image: Union[Image, str], transforms: v2.Compose, device: str, conf_thresh: float = 0.7
) -> np.ndarray:
    input_img = image.resize((640, 640))
    input_tens = transforms(input_img)
    input_tens = input_tens.to(device)
    if len(input_tens.shape) == 3:
        input_tens = input_tens.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tens)
        preds = post_process_yolo_preds(output, image, conf_thres=conf_thresh)
    if len(preds) == 0:
        return []
    return preds[0]


def generate_person_bboxes(
    model: SSD,
    image: Union[Image, str],
    transforms: v2.Compose,
    device: str,
    detect_threshold: int = 0.5,
) -> list:
    input_tens = transforms(image)
    input_tens = input_tens.to(device)
    if len(input_tens.shape) == 3:
        input_tens = input_tens.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tens)
        bxs = output[0]["boxes"]
        scores = output[0]["scores"]

        # Filter boxes by scores and labels
        idx = torch.where(scores > detect_threshold)
        classes = output[0]["labels"][idx]
        all_boxes = bxs[idx]
        boxes = [
            box for box, class_ix in zip(all_boxes, classes) if class_ix.item() == 1
        ]
        if len(boxes) == 0:
            return []
        boxes = torch.stack(boxes)
        return boxes.tolist()


def generate_keypoints(
    model: KeypointRCNN,
    image: Union[Image, str],
    transforms: v2.Compose,
    device,
    detect_threshold=0.9,
    inference: bool = False
) -> torch.Tensor:
    input_tens = transforms(image)
    input_tens = input_tens.to(device)
    if len(input_tens.shape) == 3:
        input_tens = input_tens.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tens)

    # Filter out Keypoints for Objects w/ low prob of being human
    scores = output[0]["scores"]
    kpts = output[0]["keypoints"]
    idx = torch.where(scores > detect_threshold)
    keypoints = kpts[idx]

    # Keep only upper body keypoints
    keypoints = keypoints[:, :11, :]
    if inference: 
        return keypoints
    keypoints = keypoints[keypoints[:, 0, 0].argsort()]
    return keypoints
