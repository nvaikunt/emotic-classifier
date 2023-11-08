from argparse import Namespace
from dataset.datasets import KeyPointDataset, FullImageDataset
from models.full_models import FullImageModel, KeypointsModel, OnePassModel
import math
import torch.nn as nn
from typing import List


def get_hidden_layer_sizes(start_val, num_layers, strategy="bottleneck"):
    hidden_layers = []
    for i in range(num_layers):
        if strategy == "bottleneck":
            current_layer_sz = start_val // (i + 1)
        elif strategy == "expand":
            current_layer_sz = int(start_val * math.pow(1.1, (i + 1)))
        else:
            current_layer_sz = start_val
        hidden_layers.append(current_layer_sz)
    return hidden_layers


def load_model(model_args: Namespace) -> nn.Module:
    if model_args.model_type == "keypoints":
        model = KeypointsModel(
            num_hidden_fusion_layers=model_args.num_fusion_layers,
            dropout_p=model_args.dropout,
            hidden_size_const=model_args.hidden_sz_const,
            post_concat_feat_sz=model_args.post_concat_feat_sz,
        )
    else:
        if model_args.hidden_sz_strat == "constant":
            hidden_size_list = None
        else:
            hidden_size_list = get_hidden_layer_sizes(
                model_args.hidden_sz_const,
                model_args.num_fusion_layers,
                strategy=model_args.hidden_sz_strat,
            )
        if model_args.model_type == "full_image":
            model = FullImageModel(
                num_hidden_fusion_layers=model_args.num_fusion_layers,
                dropout_p=model_args.dropout,
                hidden_sizes=hidden_size_list,
                hidden_size_const=model_args.hidden_sz_const,
                post_concat_feat_sz=model_args.post_concat_feat_sz,
            )
        else:
            model = OnePassModel(
                num_hidden_layers=model_args.num_fusion_layers,
                dropout_p=model_args.dropout,
                hidden_sizes=hidden_size_list,
                hidden_size_const=model_args.hidden_sz_const,
            )
    return model


def load_dataset(
    model_type: str,
    split: str,
    annotation_csv: str,
    data_folder: str,
    preproccesed_feat_path: str,
    device: str,
):
    yolo_model_cfg = "./models/yolov5n.yaml"
    yolo_model_pth = "./model_weights/pretrained/yolov5n-face_new.pt"
    if preproccesed_feat_path == "":
        preprocess_feats = False
        preproccesed_feat_path = None
    else:
        preprocess_feats = True
    if model_type == "keypoints":
        dataset = KeyPointDataset(
            annotation_csv=annotation_csv,
            data_folder=data_folder,
            yolo_model_cfg=yolo_model_cfg,
            yolo_model_path=yolo_model_pth,
            preprocess_feats=preprocess_feats,
            preprocessed_feat_path=preproccesed_feat_path,
            split=split,
            model_device=device,
        )
    else:
        dataset = FullImageDataset(
            annotation_csv=annotation_csv,
            data_folder=data_folder,
            yolo_model_cfg=yolo_model_cfg,
            yolo_model_path=yolo_model_pth,
            preprocess_feats=preprocess_feats,
            preprocessed_feat_path=preproccesed_feat_path,
            split=split,
            model_device=device,
        )
    return dataset


def filter_model_params(model: nn.Module, modules_to_filter: List = None):
    if modules_to_filter is None:
        modules_to_filter = ["regression.weight", "regression.bias"]
    filtered_params = list(
        filter(lambda kv: kv[0] in modules_to_filter, model.named_parameters())
    )
    base_params = list(
        filter(lambda kv: kv[0] not in modules_to_filter, model.named_parameters())
    )
    filtered_params = [param for _, param in filtered_params]
    base_params = [param for _, param in base_params]
    return filtered_params, base_params
