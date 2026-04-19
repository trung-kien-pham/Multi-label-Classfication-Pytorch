import os
import csv
import torch
import argparse
from PIL import Image
import torch.nn as nn
from model import build_model
from data import build_transforms
from typing import List, Tuple, Optional


def load_checkpoint(ckpt_path: str, device: torch.device) -> dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def load_model_from_checkpoint(
    ckpt_path: str,
    image_size: int,
    threshold_override: Optional[float],
    device: torch.device
):
    ckpt = load_checkpoint(ckpt_path, device)

    model_name = ckpt["model_name"]
    num_classes = ckpt["num_classes"]
    threshold = ckpt["threshold"] if threshold_override is None else threshold_override

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, eval_transform = build_transforms(image_size=image_size)

    return model, eval_transform, threshold, num_classes, ckpt


@torch.no_grad()
def predict_one_image(
    model: nn.Module,
    image_path: str,
    transform,
    threshold: float,
    device: torch.device
) -> Tuple[List[int], List[float]]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits)[0]

    probs_list = probs.detach().cpu().tolist()
    pred_labels = [i for i, p in enumerate(probs_list) if p >= threshold]

    return pred_labels, probs_list


def list_images_in_folder(folder: str, exts=None) -> List[str]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = []
    for name in os.listdir(folder):
        full_path = os.path.join(folder, name)
        if os.path.isfile(full_path):
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                image_paths.append(full_path)

    image_paths.sort()
    return image_paths


def format_topk_probs(probs: List[float], top_k: int) -> List[Tuple[int, float]]:
    pairs = list(enumerate(probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def print_prediction(
    image_path: str,
    pred_labels: List[int],
    probs: List[float],
    top_k: int = 10
):
    print(f"\nImage: {image_path}")

    if len(pred_labels) == 0:
        print("Predicted labels: []")
        print("No classes exceed the threshold.")
    else:
        print(f"Predicted labels: {pred_labels}")

    top_pairs = format_topk_probs(probs, top_k=top_k)

    print(f"Top-{top_k} classes:")
    for cls_id, score in top_pairs:
        print(f"  class {cls_id}: {score:.6f}")


def save_predictions_to_csv(results: List[dict], csv_path: str):
    out_dir = os.path.dirname(csv_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "filename",
        "pred_labels",
        "top1_class",
        "top1_score",
        "all_probs",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model, transform, threshold, num_classes, ckpt = load_model_from_checkpoint(
        ckpt_path=args.ckpt_path,
        image_size=args.image_size,
        threshold_override=args.threshold,
        device=device
    )

    print(f"Loaded model_name  : {ckpt['model_name']}")
    print(f"Loaded num_classes : {num_classes}")
    print(f"Using threshold    : {threshold:.4f}")

    if args.image_path is not None:
        pred_labels, probs = predict_one_image(
            model=model,
            image_path=args.image_path,
            transform=transform,
            threshold=threshold,
            device=device
        )
        print_prediction(
            image_path=args.image_path,
            pred_labels=pred_labels,
            probs=probs,
            top_k=args.top_k
        )

    if args.image_dir is not None:
        if not os.path.isdir(args.image_dir):
            raise NotADirectoryError(f"Not a valid directory: {args.image_dir}")

        image_paths = list_images_in_folder(args.image_dir)
        print(f"\nFound {len(image_paths)} images in the directory.")

        results = []

        for image_path in image_paths:
            pred_labels, probs = predict_one_image(
                model=model,
                image_path=image_path,
                transform=transform,
                threshold=threshold,
                device=device
            )

            top_pairs = format_topk_probs(probs, top_k=1)
            top1_class, top1_score = top_pairs[0]

            results.append({
                "filename": os.path.basename(image_path),
                "pred_labels": "|".join(map(str, pred_labels)) if len(pred_labels) > 0 else "",
                "top1_class": top1_class,
                "top1_score": f"{top1_score:.6f}",
                "all_probs": "|".join([f"{p:.6f}" for p in probs]),
            })

            if args.print_each:
                print_prediction(
                    image_path=image_path,
                    pred_labels=pred_labels,
                    probs=probs,
                    top_k=args.top_k
                )

        if args.output_csv is not None:
            save_predictions_to_csv(results, args.output_csv)
            print(f"\nSaved prediction CSV to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict multi-label classification from trained checkpoint"
    )

    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--image_size", type=int, default=384, help="Resize size for prediction")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold from checkpoint")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top classes to display")
    parser.add_argument("--cpu", action="store_true", help="Force running on CPU")

    parser.add_argument("--image_path", type=str, default=None, help="Predict one image")
    parser.add_argument("--image_dir", type=str, default=None, help="Predict all images in a folder")

    parser.add_argument("--output_csv", type=str, default=None, help="Path to save prediction CSV")
    parser.add_argument("--print_each", action="store_true", help="Print each image result when using --image_dir")

    args = parser.parse_args()

    if args.image_path is None and args.image_dir is None:
        raise ValueError("You must provide either --image_path or --image_dir")

    main(args)
