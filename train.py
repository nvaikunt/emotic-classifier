import torch
import torchvision
import os
from typing import Tuple
import PIL.Image as PIL_Image
from PIL.Image import Image
from torchvision.transforms import v2
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.utils import draw_keypoints
from torchvision.utils import draw_bounding_boxes
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from models.yolo import Model
from yolo_utils.general import non_max_suppression_face

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def scale_box(img2_size, box, img1_size):
    h1, w1 = img1_size
    h2, w2 = img2_size
    x_scale = w1 / w2
    y_scale = h1 / h2

    box[:, [0, 2]] *= x_scale
    box[:, [1, 3]] *= y_scale
    return box


def post_process_yolo_preds(
    preds: list, orig_image: Image, model_resolution: Tuple = (640, 640)
) -> np.ndarray:
    photo = orig_image
    preds = non_max_suppression_face(preds[0])[0]
    if preds.shape[0] == 0:
        return preds
    preds = [pred.cpu()[:4] for pred in preds]
    preds = scale_box(
        model_resolution, torch.stack(preds, dim=0), (photo.height, photo.width)
    ).round()
    return preds.numpy()


if __name__ == "__main__":
    image_dir = "aux_data"

    to_input = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)])
    to_img_tens = v2.PILToTensor()
    # model = ssdlite320_mobilenet_v3_large(
    #    weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    # )
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    # model = Model(cfg="models/yolov5n.yaml")
    # model.load_state_dict(torch.load("model_weights/pretrained/yolov5n-face_new.pt", map_location=torch.device('cpu')))
    connect_skeleton = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
    ]
    model.eval()
    with torch.no_grad():
        for filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, filename)
            curr_image = PIL_Image.open(image_path)
            # new_img_size = curr_image.resize((640, 640))
            img_tensor = to_img_tens(curr_image)
            img_tensor_float = to_input(curr_image).unsqueeze(0)

            # img_tensor_float = to_input(new_img_size).unsqueeze(0)
            output = model(img_tensor_float)
            # bxs = output[0]["boxes"]

            # preds = post_process_yolo_preds(output, curr_image)
            # res = draw_bounding_boxes(img_tensor, torch.tensor(preds))
            scores = output[0]["scores"]
            kpts = output[0]["keypoints"]

            detect_threshold = 0.9
            idx = torch.where(scores > detect_threshold)
            # print(output[0]["labels"][idx])
            # boxes = bxs[idx]
            # res = draw_bounding_boxes(img_tensor, boxes)
            keypoints = kpts[idx]
            keypoints = keypoints[:, :11, :]
            # print(keypoints.shape)
            res = draw_keypoints(
                img_tensor,
                keypoints,
                colors="blue",
                radius=3,
                connectivity=connect_skeleton,
            )
            show(res)
            plt.show()
