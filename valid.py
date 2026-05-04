import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from model import build_model
import matplotlib.pyplot as plt
from loss import BCEWithLogitsLoss
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
from data import  MultiLabelImageDataset, check_required_columns, build_transforms
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelAccuracy, MultilabelF1Score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_eval_metrics(num_classes: int, threshold: float, device: torch.device):
    loss_metric = MeanMetric().to(device)

    precision_metric = MultilabelPrecision(
        num_labels=num_classes,
        threshold=threshold,
        average="macro"
    ).to(device)

    recall_metric = MultilabelRecall(
        num_labels=num_classes,
        threshold=threshold,
        average="macro"
    ).to(device)

    accuracy_metric = MultilabelAccuracy(
        num_labels=num_classes,
        threshold=threshold,
        average="macro"
    ).to(device)

    f1_metric = MultilabelF1Score(
        num_labels=num_classes,
        threshold=threshold,
        average="macro"
    ).to(device)

    return loss_metric, precision_metric, recall_metric, accuracy_metric, f1_metric

@torch.no_grad()
def evaluate_and_collect(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    threshold: float = 0.5,
    desc: str = "Valid"
):
    model.eval()

    loss_metric, precision_metric, recall_metric, accuracy_metric, f1_metric = \
        build_eval_metrics(num_classes, threshold, device)

    all_targets = []
    all_probs = []

    pbar = tqdm(loader, desc=desc, leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)

        loss_metric.update(loss)
        precision_metric.update(probs, targets.int())
        recall_metric.update(probs, targets.int())
        accuracy_metric.update(probs, targets.int())
        f1_metric.update(probs, targets.int())

        all_targets.append(targets.detach().cpu())
        all_probs.append(probs.detach().cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_score = torch.cat(all_probs, dim=0).numpy()

    metrics = {
        "loss": loss_metric.compute().item(),
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
        "accuracy": accuracy_metric.compute().item(),
        "macro_f1": f1_metric.compute().item(),
    }

    return metrics, y_true, y_score

def plot_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: str,
    class_names=None
):
    os.makedirs(output_dir, exist_ok=True)

    num_classes = y_true.shape[1]
    ap_list = []

    for c in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, c], y_score[:, c])

        try:
            ap = average_precision_score(y_true[:, c], y_score[:, c])
        except ValueError:
            ap = float("nan")

        ap_list.append(ap)

        class_label = class_names[c] if class_names is not None else f"class_{c}"

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AP = {ap:.4f}" if not np.isnan(ap) else "AP = nan")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {class_label}")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"pr_curve_class_{c}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, c], y_score[:, c])

        class_label = class_names[c] if class_names is not None else f"class_{c}"
        ap = ap_list[c]

        if np.isnan(ap):
            label = f"{class_label} (AP=nan)"
        else:
            label = f"{class_label} (AP={ap:.3f})"

        plt.plot(recall, precision, label=label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for All Classes")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve_all_classes.png"), dpi=200)
    plt.close()

    ap_df = pd.DataFrame({
        "class_id": list(range(num_classes)),
        "AP": ap_list
    })
    ap_df.to_csv(os.path.join(output_dir, "per_class_ap.csv"), index=False)

    valid_ap = [x for x in ap_list if not np.isnan(x)]
    map_value = float(np.mean(valid_ap)) if len(valid_ap) > 0 else float("nan")

    return ap_list, map_value

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    ckpt = torch.load(args.ckpt_path, map_location=device)

    ckpt_model_name = ckpt["model_name"]
    ckpt_num_classes = ckpt["num_classes"]
    ckpt_threshold = ckpt.get("threshold", 0.4)

    threshold = args.threshold if args.threshold is not None else ckpt_threshold

    print(f"Checkpoint model_name : {ckpt_model_name}")
    print(f"Checkpoint num_classes: {ckpt_num_classes}")
    print(f"Using threshold       : {threshold:.4f}")

    valid_df = pd.read_csv(args.csv_path)
    check_required_columns(valid_df, "valid.csv", args.id_col, args.label_col)

    num_classes = ckpt_num_classes if args.num_classes is None else args.num_classes
    print(f"Validation samples: {len(valid_df)}")

    _, eval_transform = build_transforms(args.image_size)

    valid_dataset = MultiLabelImageDataset(
        df=valid_df,
        image_dir=args.image_dir,
        num_classes=num_classes,
        transform=eval_transform,
        id_col=args.id_col,
        label_col=args.label_col,
        image_ext=args.image_ext,
        label_sep=args.label_sep
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model_name = args.model_name if args.model_name is not None else ckpt_model_name

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    criterion = BCEWithLogitsLoss()

    metrics, y_true, y_score = evaluate_and_collect(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        threshold=threshold,
        desc="Valid"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    ap_list, map_value = plot_pr_curves(
        y_true=y_true,
        y_score=y_score,
        output_dir=args.output_dir,
        class_names=None
    )

    print("\n========== Validation Results ==========")
    print(f"Loss : {metrics['loss']:.4f}")
    print(f"PR   : {metrics['precision']:.4f}")
    print(f"RC   : {metrics['recall']:.4f}")
    print(f"ACC  : {metrics['accuracy']:.4f}")
    print(f"M-F1 : {metrics['macro_f1']:.4f}")
    print(f"mAP  : {map_value:.4f}")

    print(f"\nSaved PR curves to: {args.output_dir}")
    print("Files:")
    print("- pr_curve_all_classes.png")
    print("- pr_curve_class_0.png, pr_curve_class_1.png, ...")
    print("- per_class_ap.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate multi-label model and plot PR curves per class"
    )

    parser.add_argument("--csv_path", type=str, required=True, help="Path to validation/test CSV")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--output_dir", type=str, default="valid_outputs", help="Directory to save PR curves")

    parser.add_argument("--id_col", type=str, default="ID", help="Column name containing image IDs")
    parser.add_argument("--label_col", type=str, default="Label", help="Column name containing labels")

    parser.add_argument("--image_ext", type=str, default="", help="Image extension, e.g. .png or .jpg")
    parser.add_argument("--label_sep", type=str, default="|", help="Separator for multiple labels")

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["resnet18", "resnet50", "efficientnet_b0", "efficientnet_v2_s", "convnext_small", "swin_t", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"],
        help="Optional override model name"
    )
    parser.add_argument("--num_classes", type=int, default=None, help="Optional override num_classes")
    parser.add_argument("--image_size", type=int, default=384, help="Image resize size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for PR/RC/ACC/F1 metrics")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    args = parser.parse_args()
    main(args)