import argparse
from argparse import Namespace
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import wandb
from tqdm import tqdm
import os
from typing import Tuple
import PIL.Image as PIL_Image
from PIL.Image import Image

import numpy as np
import torchvision.transforms.functional as F

from dataset.datasets import KeyPointDataset, FullImageDataset
from models.full_models import FullImageModel, KeypointsModel, OnePassModel
from utils.model_utils import get_hidden_layer_sizes
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ExponentialLR


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
                num_hidden_fusion_layers=model_args.num_fusion_layers,
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


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    epoch: int,
    device: str,
    model_type: str,
):
    model.eval()
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        print(f"Starting Validation, currently in Epoch {epoch}")
        for _, val_batch in tqdm(enumerate(dataloader)):
            # Put Tensors on Device
            body_tensor, face_tensor, labels = val_batch
            body_tensor.to(device)
            face_tensor.to(device)
            labels.to(device)

            # Get outputs

            if model_type == "single_body_image":
                outputs = model(body_tensor)
            elif model_type == "single_face_image":
                outputs = model(face_tensor)
            else:
                outputs = model(body_tensor, face_tensor)

            # Loss and Step
            loss = mse_loss(outputs, labels)
            val_loss += loss.item()
    model.train()
    return val_loss / len(dataloader)


def train(
    train_args: Namespace,
    optim_args: Namespace,
    model_args: Namespace,
    data_folder: str,
    preproccesed_train_path=str,
    preproccesed_eval_path=str,
    device: str = "cpu",
):
    # Load Models
    model = load_model(model_args=model_args)
    model.to(device)
    model.train()

    # Load Datasets
    train_dataset = load_dataset(
        model_type=model_args.model_type,
        split="train",
        annotation_csv=os.path.join(data_folder, "emotic_pre", "train.csv"),
        data_folder=data_folder,
        preproccesed_feat_path=preproccesed_train_path,
        device=device,
    )
    print(train_dataset[1])

    val_dataset = load_dataset(
        model_type=model_args.model_type,
        split="val",
        annotation_csv=os.path.join(data_folder, "emotic_pre", "val.csv"),
        data_folder=data_folder,
        preproccesed_feat_path=preproccesed_eval_path,
        device=device,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_args.batch_sz, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=train_args.batch_sz, shuffle=True
    )

    # Intialize Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=optim_args.lr)
    if optim_args.sched_type == "linear":
        scheduler = LinearLR(
            optimizer,
            start_factor=optim_args.start_factor,
            end_factor=optim_args.end_factor,
            total_iters=train_args.num_epochs,
        )
    else:
        scheduler = ExponentialLR(optimizer, gamma=optim_args.gamma)
    mse_loss = nn.MSELoss()
    # Start Logging
    
    run = wandb.init(
        project=f"EMOTIC Based Visual Confusion Detection",
        config={
            "model_type": model_args.model_type,
            "learning_rate": optim_args.lr,
            "epochs": train_args.num_epochs,
            "batch_size": train_args.batch_sz,
            "optim_type": optim_args.sched_type,
        },
    )
    
    train_loss = 0

    # Begin Loop
    for epoch in range(train_args.num_epochs):
        print(f"Starting Training for Epoch {epoch}!")
        for step, train_batch in tqdm(enumerate(train_dataloader)):
            # Put Tensors on Device
            body_tensor, face_tensor, labels = train_batch
            body_tensor = body_tensor.to(device)
            face_tensor = face_tensor.to(device)
            labels = labels.to(device)

            # Get outputs
            optimizer.zero_grad()
            if model_args.model_type == "single_body_image":
                outputs = model(body_tensor)
            elif model_args.model_type == "single_face_image":
                outputs = model(face_tensor)
            else:
                outputs = model(body_tensor, face_tensor)

            # Loss and Step
            loss = mse_loss(outputs, labels)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Logging and Eval if Using Step Wise Strategy
            if train_args.log_strat == "steps" and step % train_args.log_steps == 0:
                avg_loss = train_loss / train_args.log_steps
                wandb.log({"train_mse": avg_loss})
                train_loss = 0
            if train_args.eval_strat == "steps" and step % train_args.eval_steps == 0:
                val_loss = validate(
                    model,
                    val_dataloader,
                    epoch,
                    device,
                    model_type=model_args.model_type,
                )
                wandb.log({"val_mse": val_loss})
        # Epoch Logging and Eval
        if train_args.log_strat == "epoch":
            avg_loss = train_loss / len(train_dataloader)
            wandb.log({"train_mse": avg_loss})
            train_loss = 0
        if train_args.eval_strat == "epoch":
            val_loss = validate(
                model, val_dataloader, epoch, device, model_type=model_args.model_type
            )
            wandb.log({"val_mse": val_loss})
        # Anneal LR
        scheduler.step()

    print(f"Training Concluded! Saving model to folder {train_args.save_folder}")
    save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(
        train_args.save_folder, f"{model_args.model_type}_{save_time}"
    )
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMOTIC Model Training")

    # Model Args
    parser.add_argument("--model_type", type=str, default="keypoint")
    parser.add_argument("--num_fusion_layers", type=int, default=3)
    parser.add_argument("--hidden_sz_const", type=int, default=512)
    parser.add_argument("--post_concat_feat_sz", type=int, default=512)
    parser.add_argument("--hidden_size_strategy", type=str, default="constant")
    parser.add_argument("--dropout", type=float, default=0.3)

    # Optimization Args
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_sched_type", type=str, default="linear")
    parser.add_argument("--linear_start", type=float, default=1.0)
    parser.add_argument("--linear_end", type=float, default=0.1)
    parser.add_argument("--exp_gamma", type=float, default=0.8)

    # Train Args
    parser.add_argument("--num_epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=float, default=8)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--eval_steps", type=int, default=700)
    parser.add_argument("--log_strategy", type=str, default="steps")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_folder", type=str, default="./saved_models")

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--emotic_folder", type=str, default="./emotic")
    parser.add_argument("--preprocessed_train_path", type=str, default="")
    parser.add_argument("--preprocessed_eval_path", type=str, default="")

    args = parser.parse_args()
    model_args = Namespace(
        model_type=args.model_type,
        num_fusion_layers=args.num_fusion_layers,
        hidden_sz_const=args.hidden_sz_const,
        post_concat_feat_sz=args.post_concat_feat_sz,
        hidden_sz_strat=args.hidden_size_strategy,
        dropout=args.dropout,
    )

    optim_args = Namespace(
        lr=args.lr,
        sched_type=args.lr_sched_type,
        start_factor=args.linear_start,
        end_factor=args.linear_end,
        gamma=args.exp_gamma,
    )

    train_args = Namespace(
        num_epochs=args.num_epochs,
        batch_sz=args.batch_size,
        eval_strat=args.eval_strategy,
        eval_steps=args.eval_steps,
        log_strat=args.log_strategy,
        log_steps=args.log_steps,
        save_folder=args.save_folder,
    )

    train(
        train_args=train_args,
        optim_args=optim_args,
        model_args=model_args,
        data_folder=args.emotic_folder,
        preproccesed_train_path=args.preprocessed_train_path,
        preproccesed_eval_path=args.preprocessed_eval_path,
        device=args.device,
    )
