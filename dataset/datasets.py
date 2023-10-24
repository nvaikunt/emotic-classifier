import torch
from torch.utils.data import Dataset
import os
from typing import Tuple, List
import pickle
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
from models.yolo import Model

from utils.data_utils import load_and_clean_data, get_cropped_image
from utils.model_utils import generate_face_bbox, generate_keypoints


class GenericDataset(Dataset):
    def __init__(
        self,
        annotation_csv: str,
        data_folder: str,
        preprocess_feats: bool = False,
        preprocessed_feat_path: str = None,
        model_device: str = "cpu",
        yolo_model_path: str = "model_weights/pretrained/yolov5n-face_new.pt",
        yolo_model_cfg: str = "models/yolov5n.yaml",
    ):
        super(Dataset, self).__init__()
        # Init class vars
        self.model_device = model_device
        self.preprocess_feats = preprocess_feats
        self.preprocess_feat_path = preprocessed_feat_path
        self.data_folder = data_folder

        # Get annotation data ready in unprocessed form
        self.dataset = load_and_clean_data(
            csv_file=annotation_csv, data_folder=data_folder, return_dict_list=True
        )

        if self.preprocess_feat_path is not None:
            self.preprocess_feats = True
            with open(self.preprocess_feat_path, "rb") as processed_feat_list:
                self.dataset = pickle.load(processed_feat_list)

        # Initialize feature extractor 1 -> face extraction
        self.face_detector = Model(cfg=yolo_model_cfg)
        self.face_detector.load_state_dict(
            torch.load(yolo_model_path, map_location=torch.device("cpu"))
        )
        self.face_detector.eval()
        self.face_detector.to(self.model_device)

        self.tensor_transforms = v2.Compose(
            [v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)]
        )

    @staticmethod
    def get_poi_from_img_path(record: dict) -> Image:
        poi_bbox = record["BBox"]
        poi_image = get_cropped_image(record["img_path"], poi_bbox)
        return poi_image

    def get_face_tensor_from_img_path(self, full_image: Image) -> torch.Tensor:
        face_bbox = generate_face_bbox(
            self.face_detector, full_image, self.tensor_transforms
        )
        face_image = get_cropped_image(full_image, face_bbox)
        return self.tensor_transforms(face_image)

    def __len__(self):
        return len(self.dataset)


class KeyPointDataset(GenericDataset):
    def __init__(
        self,
        annotation_csv: str,
        data_folder: str,
        preprocess_feats: bool = False,
        preprocessed_feat_path: str = None,
        model_device: str = "cpu",
        yolo_model_path: str = "model_weights/pretrained/yolov5n-face_new.pt",
        yolo_model_cfg: str = "models/yolov5n.yaml",
        keypoint_detect_threshold: float = 0.9,
    ):
        super().__init__(
            annotation_csv,
            data_folder,
            preprocess_feats,
            preprocessed_feat_path,
            model_device,
            yolo_model_path,
            yolo_model_cfg,
        )
        self.keypoint_detect_threshold = keypoint_detect_threshold
        # Initialize feature extractor 2 -> keypoint extraction
        self.keypoint_model = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.keypoint_model.eval()
        self.keypoint_model.to(self.model_device)

        if self.preprocess_feats and self.preprocess_feat_path is None:
            self.dataset = self.preprocess()

    def preprocess(self) -> List[dict]:
        for record in self.dataset:
            poi_image = self.get_poi_from_img_path(record)
            # Get Image Tensor of Face
            record["face_tensor"] = self.get_face_tensor_from_img_path(poi_image)
            # Get
            record["keypoints"] = generate_keypoints(
                self.keypoint_model,
                poi_image,
                self.tensor_transforms,
                detect_threshold=self.keypoint_detect_threshold,
            )
        return self.dataset

    def __getitem__(self, idx):
        record = self.dataset[idx]
        labels = torch.Tensor(record["Continuous_Labels"])
        if self.preprocess_feats:
            face_img_tensor = record["face_tensor"]
            keypoint_tensor = record["keypoints"]
        else:
            poi_image = self.get_poi_from_img_path(record)
            face_img_tensor = self.get_face_tensor_from_img_path(poi_image)
            keypoint_tensor = generate_keypoints(
                self.keypoint_model,
                poi_image,
                self.tensor_transforms,
                detect_threshold=self.keypoint_detect_threshold,
            )
        return face_img_tensor, keypoint_tensor, labels

