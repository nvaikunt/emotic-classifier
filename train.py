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
from utils.train_utils import load_dataset, load_model, filter_model_params
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ExponentialLR


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    epoch: int,
    device: str,
    model_type: str,
    clamp: bool,
):
    model.eval()
    mse_loss = nn.MSELoss()
    avg_losses = []
    val_loss = 0
    with torch.no_grad():
        print(f"Starting Validation, currently in Epoch {epoch}")
        for step, val_batch in tqdm(enumerate(dataloader)):
            # Put Tensors on Device
            body_tensor, face_tensor, labels = val_batch
            body_tensor = body_tensor.to(device)
            face_tensor = face_tensor.to(device)
            labels = labels.to(device)

            # Get outputs

            if model_type == "single_body_image":
                outputs = model(body_tensor)
            elif model_type == "single_face_image":
                outputs = model(face_tensor)
            else:
                outputs = model(body_tensor, face_tensor)
            if clamp:
                outputs = torch.clamp(outputs, 1, 10)
            print(f"Output Reg {outputs}")
            print(f"Target Reg {labels}")
            # Loss and Step
            loss = mse_loss(outputs, labels)
            print(f"Loss for batch: {loss}")
            val_loss += loss.item()
            if step % 10 == 0:
                avg_loss = val_loss / 10
                val_loss = 0
                avg_losses.append(avg_loss)
    model.train()
    return sum(avg_losses) / len(avg_losses)


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
    reg_head_params, backbone_params = filter_model_params(model)
    param_config = [
        {"params": backbone_params},
        {"params": reg_head_params, "lr": optim_args.regression_lr},
    ]
    optimizer = optim.Adam(param_config, lr=optim_args.lr)
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
            "regression_learning_rate": optim_args.regression_lr,
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
            if train_args.clamp:
                outputs = torch.clamp(outputs, 1, 10)
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
                    clamp=train_args.clamp,
                )
                wandb.log({"val_mse": val_loss})
        # Epoch Logging and Eval
        if train_args.log_strat == "epoch":
            avg_loss = train_loss / len(train_dataloader)
            wandb.log({"train_mse": avg_loss})
            train_loss = 0
        if train_args.eval_strat == "epoch":
            val_loss = validate(
                model,
                val_dataloader,
                epoch,
                device,
                model_type=model_args.model_type,
                clamp=train_args.clamp,
            )
            wandb.log({"val_mse": val_loss})
        # Anneal LR
        scheduler.step()

    print(f"Training Concluded! Saving model to folder {train_args.save_folder}")
    os.makedirs(train_args.save_folder, exist_ok=True)
    save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(
        train_args.save_folder, f"{model_args.model_type}_{save_time}.pt"
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
    parser.add_argument("--regression_lr", type=float, default=1e-2)
    parser.add_argument("--lr_sched_type", type=str, default="linear")
    parser.add_argument("--linear_start", type=float, default=1.0)
    parser.add_argument("--linear_end", type=float, default=0.1)
    parser.add_argument("--exp_gamma", type=float, default=0.8)

    # Train Args
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--eval_steps", type=int, default=700)
    parser.add_argument("--log_strategy", type=str, default="steps")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_folder", type=str, default="./saved_models")
    parser.add_argument("--clamp", action="store_true")

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
        regression_lr=args.regression_lr,
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
        clamp=args.clamp,
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
