import os
import copy
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelF1Score
from data import (
    MultiLabelImageDataset,
    infer_num_classes_from_dfs,
    check_required_columns,
    build_transforms,
)
from loss import BCEWithLogitsLoss
from model import build_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_epoch_metrics(num_classes: int, threshold: float, device: torch.device):
    loss_metric = MeanMetric().to(device)
    f1_metric = MultilabelF1Score(
        num_labels=num_classes,
        threshold=threshold,
        average="macro"
    ).to(device)

    return loss_metric, f1_metric


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    threshold: float = 0.5
) -> Tuple[float, float]:
    model.train()

    loss_metric, f1_metric = build_epoch_metrics(num_classes, threshold, device)
    pbar = tqdm(loader, desc="Train", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)

        loss_metric.update(loss.detach())
        f1_metric.update(probs.detach(), targets.int())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = loss_metric.compute().item()
    epoch_f1 = f1_metric.compute().item()

    return epoch_loss, epoch_f1


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    threshold: float = 0.5,
    desc: str = "Eval"
) -> Tuple[float, float]:
    model.eval()

    loss_metric, f1_metric = build_epoch_metrics(num_classes, threshold, device)
    pbar = tqdm(loader, desc=desc, leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        loss_metric.update(loss)
        f1_metric.update(probs, targets.int())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = loss_metric.compute().item()
    epoch_f1 = f1_metric.compute().item()

    return epoch_loss, epoch_f1


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    test_df = None
    if args.test_csv is not None and str(args.test_csv).strip() != "":
        test_df = pd.read_csv(args.test_csv)

    check_required_columns(train_df, "train.csv", args.id_col, args.label_col)
    check_required_columns(val_df, "val.csv", args.id_col, args.label_col)

    if test_df is not None:
        check_required_columns(test_df, "test.csv", args.id_col, args.label_col)

    if args.num_classes is None:
        dfs_for_infer = [train_df, val_df]
        if test_df is not None:
            dfs_for_infer.append(test_df)

        num_classes = infer_num_classes_from_dfs(
            dfs=dfs_for_infer,
            label_col=args.label_col,
            sep=args.label_sep
        )
        print(f"Inferred num_classes = {num_classes}")
    else:
        num_classes = args.num_classes
        print(f"Using provided num_classes = {num_classes}")

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    if test_df is not None:
        print(f"Test samples:  {len(test_df)}")

    train_transform, eval_transform = build_transforms(args.image_size)

    train_dataset = MultiLabelImageDataset(
        df=train_df,
        image_dir=args.image_dir,
        num_classes=num_classes,
        transform=train_transform,
        id_col=args.id_col,
        label_col=args.label_col,
        image_ext=args.image_ext,
        label_sep=args.label_sep
    )

    val_dataset = MultiLabelImageDataset(
        df=val_df,
        image_dir=args.image_dir,
        num_classes=num_classes,
        transform=eval_transform,
        id_col=args.id_col,
        label_col=args.label_col,
        image_ext=args.image_ext,
        label_sep=args.label_sep
    )

    test_dataset = None
    if test_df is not None:
        test_dataset = MultiLabelImageDataset(
            df=test_df,
            image_dir=args.image_dir,
            num_classes=num_classes,
            transform=eval_transform,
            id_col=args.id_col,
            label_col=args.label_col,
            image_ext=args.image_ext,
            label_sep=args.label_sep
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    model = build_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=not args.no_pretrained
    )
    model = model.to(device)

    criterion = BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    best_val_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")

        train_loss, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            threshold=args.threshold
        )

        val_loss, val_f1 = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            threshold=args.threshold,
            desc="Val"
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Macro F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save({
                "model_state_dict": best_model_wts,
                "model_name": args.model_name,
                "num_classes": num_classes,
                "threshold": args.threshold,
                "id_col": args.id_col,
                "label_col": args.label_col,
                "image_ext": args.image_ext,
                "label_sep": args.label_sep
            }, best_model_path)

            print(f"Saved best model to: {best_model_path}")

    history_df = pd.DataFrame(history)
    history_csv = os.path.join(args.output_dir, "train_log.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"\nSaved training log to: {history_csv}")

    model.load_state_dict(best_model_wts)
    print(f"Best Val Macro F1: {best_val_f1:.4f}")

    if test_loader is not None:
        test_loss, test_f1 = evaluate_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            threshold=args.threshold,
            desc="Test"
        )

        print(f"Test Loss: {test_loss:.4f} | Test Macro F1: {test_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multi-label image classification with train/val/test CSV"
    )

    parser.add_argument("--train_csv", type=str, required=True, help="The path to train.csv")
    parser.add_argument("--val_csv", type=str, required=True, help="The path to val.csv")
    parser.add_argument("--test_csv", type=str, default=None, help="The path to test.csv")
    parser.add_argument("--image_dir", type=str, required=True, help="The directory containing images")
    parser.add_argument("--output_dir", type=str, default="outputs", help="The directory to save results")

    parser.add_argument("--id_col", type=str, default="ID", help="The column name containing image IDs")
    parser.add_argument("--label_col", type=str, default="Label", help="The column name containing labels")

    parser.add_argument("--image_ext", type=str, default="", help="Image file extension, e.g., .png or .jpg")
    parser.add_argument("--label_sep", type=str, default="|", help="Separator for multiple labels")

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "efficientnet_b0", "efficientnet_v2_s", "convnext_small", "swin_t", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"],
        help="The name of the torchvision model"
    )
    parser.add_argument("--num_classes", type=int, default=None, help="Total number of classes. If empty, it will be inferred")
    parser.add_argument("--no_pretrained", action="store_true", help="Do not use pretrained weights")

    parser.add_argument("--image_size", type=int, default=384, help="Image size for resizing")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=0.4, help="Threshold for predicting labels")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)