import torch
from torch.utils.data import Dataset
import os
from typing import Tuple, List, Optional
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
import sys

sys.path.append("./")
from models.yolo import Model
from tqdm import tqdm
from utils.data_utils import load_and_clean_data, get_cropped_image
from utils.model_utils import (
    generate_face_bbox,
    generate_keypoints,
    generate_person_bboxes,
)
import datetime


class GenericDataset(Dataset):
    def __init__(
        self,
        annotation_csv: str,
        data_folder: str,
        yolo_model_path: str,
        yolo_model_cfg: str,
        preprocess_feats: bool,
        preprocessed_feat_path: Optional[bool],
        split: str,
        model_device: str,
    ):
        super(Dataset, self).__init__()
        # Init class vars
        self.model_device = model_device
        self.preprocess_feats = preprocess_feats
        self.preprocess_feat_path = preprocessed_feat_path
        self.data_folder = data_folder
        self.split = split

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

        # Initialize SSD person detector
        self.person_detector = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.person_detector.eval()
        self.person_detector.to(self.model_device)
        self.tensor_transforms = v2.Compose(
            [v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)]
        )

    @staticmethod
    def get_roi_from_img_path(record: dict) -> Image:
        poi_bbox = record["BBox"]
        poi_image = get_cropped_image(record["img_path"], poi_bbox)
        return poi_image

    def get_person_images_from_roi(self, full_image: Image) -> List[Image]:
        people_bboxes = generate_person_bboxes(
            self.person_detector, full_image, self.tensor_transforms, self.model_device
        )
        if not people_bboxes:
            return None
        people_images = [
            get_cropped_image(full_image, person_bbox) for person_bbox in people_bboxes
        ]
        return people_images

    def get_face_tensor_from_img(self, full_image: Image) -> Optional[torch.Tensor]:
        face_bbox = generate_face_bbox(
            self.face_detector, full_image, self.tensor_transforms, self.model_device
        )
        if len(face_bbox) == 0:
            return None
        face_image = get_cropped_image(full_image, face_bbox)
        face_image = face_image.resize((160, 160))
        return self.tensor_transforms(face_image)
    
    def get_full_img_tensor(self, image: Image):
        image = image.resize((160, 160))
        return self.tensor_transforms(image)

    def __len__(self):
        return len(self.dataset)

    def save_preprocessed(self):
        save_path = os.path.join(
            data_folder, "preprocessed_feats", f"{self.split}_{datetime.date.today()}"
        )
        with open(save_path, "wb") as save_file:
            pickle.dump(self.dataset, save_file)


class KeyPointDataset(GenericDataset):
    def __init__(
        self,
        annotation_csv: str,
        data_folder: str,
        yolo_model_path: str = "model_weights/pretrained/yolov5n-face_new.pt",
        yolo_model_cfg: str = "models/yolov5n.yaml",
        preprocess_feats: bool = False,
        preprocessed_feat_path: str = None,
        split: str = "train",
        model_device: str = "cpu",
        keypoint_detect_threshold: float = 0.3,
    ):
        super().__init__(
            annotation_csv,
            data_folder,
            yolo_model_path,
            yolo_model_cfg,
            preprocess_feats,
            preprocessed_feat_path,
            split,
            model_device,
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
            self.save_preprocessed()

    def preprocess(self) -> List[dict]:
        new_dataset = []
        skipped_examples = []
        for record in tqdm(self.dataset):
            roi_image = self.get_roi_from_img_path(record)
            person_images = self.get_person_images_from_roi(roi_image)
            # Get Image Tensor of Face
            for poi_image in person_images:
                try: 
                    new_record = {
                        "face_tensor": self.get_face_tensor_from_img(poi_image),
                        "keypoints": generate_keypoints(
                            self.keypoint_model,
                            poi_image,
                            self.tensor_transforms,
                            self.model_device,
                            detect_threshold=self.keypoint_detect_threshold,
                        )[0],
                        "labels": torch.Tensor(record["Continuous_Labels"]),
                    }
                    if new_record["face_tensor"] is None:
                        skipped_examples.append(record["img_path"])
                        continue
                except IndexError: 
                    # print(f"WARNING: No keypoints found for {record['img_path']}. Skipping example")
                    skipped_examples.append(record["img_path"])
                    continue
        
                # Get Keypoints
                new_dataset.append(new_record)
        print(f"Number of Skipped Images: {len(skipped_examples)}")
        print(skipped_examples[0])
        print(skipped_examples[-1])
        return new_dataset

    def __getitem__(self, idx):
        record = self.dataset[idx]

        if self.preprocess_feats:
            labels = record["labels"]
            face_img_tensor = record["face_tensor"]
            keypoint_tensor = record["keypoints"]
        else:
            labels = torch.Tensor(record["Continuous_Labels"])
            roi_image = self.get_roi_from_img_path(record)
            person_images = self.get_person_images_from_roi(roi_image)
            if person_images is not None: 
                poi_image = person_images[0]
            else: 
                poi_image = roi_image
            face_img_tensor = self.get_face_tensor_from_img(poi_image)
            if face_img_tensor is None: 
                face_img_tensor = self.get_full_img_tensor(poi_image)
            try: 
                keypoint_tensor = generate_keypoints(
                    self.keypoint_model,
                    poi_image,
                    self.tensor_transforms,
                    self.model_device,
                    detect_threshold=self.keypoint_detect_threshold,
                )[0]
            except IndexError: 
                keypoint_tensor = torch.rand((11, 3))

        return keypoint_tensor, face_img_tensor, labels


class FullImageDataset(GenericDataset):
    def __init__(
        self,
        annotation_csv: str,
        data_folder: str,
        yolo_model_path: str = "model_weights/pretrained/yolov5n-face_new.pt",
        yolo_model_cfg: str = "models/yolov5n.yaml",
        preprocess_feats: bool = False,
        preprocessed_feat_path: str = None,
        split: str = "train",
        model_device: str = "cpu",
    ):
        super().__init__(
            annotation_csv,
            data_folder,
            yolo_model_path,
            yolo_model_cfg,
            preprocess_feats,
            preprocessed_feat_path,
            split,
            model_device,
        )
        if self.preprocess_feats and self.preprocess_feat_path is None:
            self.dataset = self.preprocess()
            self.save_preprocessed()



    def preprocess(self) -> List[dict]:
        new_dataset = []
        skipped_examples = []
        for record in tqdm(self.dataset):
            roi_image = self.get_roi_from_img_path(record)
            person_images = self.get_person_images_from_roi(roi_image)
            for poi_image in person_images:
                new_record = {
                    "full_tensor": self.get_full_img_tensor(
                        poi_image
                    ),  # Get tensor of full
                    "face_tensor": self.get_face_tensor_from_img(
                        poi_image
                    ),  # Get Image Tensor of Face
                    "labels": torch.Tensor(record["Continuous_Labels"]),
                }
                if new_record["face_tensor"] is None or new_record["full_tensor"] is None:
                    skipped_examples.append(record["img_path"])
                    continue
                new_dataset.append(new_record)
        print(f"Number of Skipped Images: {len(skipped_examples)}")
        print(skipped_examples[0])
        print(skipped_examples[-1])
        return new_dataset

    def __getitem__(self, idx):
        record = self.dataset[idx]

        if self.preprocess_feats:
            labels = record["labels"]
            face_img_tensor = record["face_tensor"]
            full_img_tensor = record["full_tensor"]
        else:
            labels = torch.Tensor(record["Continuous_Labels"])
            roi_image = self.get_roi_from_img_path(record)
            person_images = self.get_person_images_from_roi(roi_image)
            if person_images is not None: 
                poi_image = person_images[0]
            else: 
                poi_image = roi_image
            face_img_tensor = self.get_face_tensor_from_img(poi_image)
            if face_img_tensor is None: 
                face_img_tensor = self.get_full_img_tensor(poi_image)
            full_img_tensor = self.get_full_img_tensor(poi_image)
        return  full_img_tensor, face_img_tensor, labels


if __name__ == "__main__":
    train_csv = "./emotic/emotic_pre/train.csv"
    val_csv = "./emotic/emotic_pre/val.csv"
    data_folder = "./emotic"
    yolo_config_full = (
        "./models/yolov5n.yaml"
    )
    yolo_weights = "./model_weights/pretrained/yolov5n-face_new.pt"


    train_keypoint_dataset = KeyPointDataset(
        annotation_csv=train_csv,
        data_folder=data_folder,
        preprocess_feats=True,
        yolo_model_cfg=yolo_config_full,
        yolo_model_path=yolo_weights,
        model_device="cuda",
        split="train",
        keypoint_detect_threshold=0.7
    )

    val_keypoint_dataset = KeyPointDataset(
        annotation_csv=val_csv,
        data_folder=data_folder,
        preprocess_feats=True,
        yolo_model_cfg=yolo_config_full,
        yolo_model_path=yolo_weights,
        model_device="cuda",
        split="val",
        keypoint_detect_threshold=0.7
    )

    train_image_dataset = FullImageDataset(
        annotation_csv=train_csv,
        data_folder=data_folder,
        preprocess_feats=True,
        yolo_model_cfg=yolo_config_full,
        yolo_model_path=yolo_weights,
        model_device="cuda",
        split="train",
    )

    val_image_dataset = FullImageDataset(
        annotation_csv=val_csv,
        data_folder=data_folder,
        preprocess_feats=True,
        yolo_model_cfg=yolo_config_full,
        yolo_model_path=yolo_weights,
        model_device="cuda",
        split="val",
    )
