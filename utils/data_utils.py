import os
from typing import Union, Tuple

import PIL.Image as PIL_Image
from PIL.Image import Image
import pandas as pd
from ast import literal_eval
from random import shuffle
import numpy as np
import torch
from yolo_utils.general import non_max_suppression_face


def create_full_filepath(parent_dir: str, subdir: str, file_name: str) -> str:
    return os.path.join(parent_dir, subdir, file_name)


def load_and_clean_data(
    csv_file: str, data_folder: str, split_vad=True, return_dict_list=False
):
    annotations_df = pd.read_csv(csv_file)
    annotations_df["img_path"] = annotations_df.apply(
        lambda x: create_full_filepath(data_folder, x.Folder, x.Filename), axis=1
    )
    annotations_df["Image Size"] = annotations_df["Image Size"].apply(literal_eval)
    annotations_df["BBox"] = annotations_df["BBox"].apply(literal_eval)
    annotations_df["Categorical_Labels"] = annotations_df["Categorical_Labels"].apply(
        literal_eval
    )
    annotations_df["Continuous_Labels"] = annotations_df["Continuous_Labels"].apply(
        literal_eval
    )
    if split_vad:
        annotations_df[["valence", "arousal", "dominance"]] = pd.DataFrame(
            annotations_df["Continuous_Labels"].tolist(), index=annotations_df.index
        )
    if return_dict_list:
        return annotations_df.to_dict("records")
    return annotations_df


def load_pictures(annotation_df: pd.DataFrame, num_pics: int = 10, random: bool = True):
    list_of_filepaths = list(
        zip(annotation_df["img_path"].tolist(), annotation_df["BBox"].tolist())
    )
    if random:
        shuffle(list_of_filepaths)
    for ix in range(num_pics):
        filepath, bounding_box = list_of_filepaths[ix]
        image = PIL_Image.open(filepath)
        cropped_image = image.crop(tuple(bounding_box))
        image.show()
        cropped_image.show()


def get_cropped_image(
    image: Union[Image, str], bbox: Union[list, torch.Tensor, np.ndarray]
) -> Image:
    if not isinstance(image, Image):
        image = PIL_Image.open(image)

    if not isinstance(bbox, list):
        bbox = bbox.tolist()
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.crop(tuple(bbox))


def scale_box(img2_size, box, img1_size):
    h1, w1 = img1_size
    h2, w2 = img2_size
    x_scale = w1 / w2
    y_scale = h1 / h2

    box[:, [0, 2]] *= x_scale
    box[:, [1, 3]] *= y_scale
    return box


def post_process_yolo_preds(
    preds: list, orig_image: Image, model_resolution: Tuple = (640, 640), conf_thres=0.7
) -> np.ndarray:
    photo = orig_image
    preds = non_max_suppression_face(preds[0], conf_thres=conf_thres)[0]
    if preds.shape[0] == 0:
        return preds
    preds = [pred.cpu()[:4] for pred in preds]
    preds = scale_box(
        model_resolution, torch.stack(preds, dim=0), (photo.height, photo.width)
    ).round()
    preds = torch.clamp(preds, min=0)
    preds = preds[preds[:, 0].argsort()]
    return preds.numpy()
