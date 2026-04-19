import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



def infer_num_classes_from_dfs(
    dfs: list[pd.DataFrame],
    label_col: str = "Label",
    sep: str = "|"
) -> int:
    max_label = -1

    for df in dfs:
        if df is None:
            continue

        for label_str in df[label_col].astype(str).values:
            parts = [x.strip() for x in label_str.split(sep) if x.strip() != ""]
            if len(parts) == 0:
                continue

            label_ids = [int(x) for x in parts]
            max_label = max(max_label, max(label_ids))

    if max_label < 0:
        raise ValueError("No valid labels were found in the CSV files.")

    return max_label + 1


def encode_multilabel(label_str: str, num_classes: int, sep: str = "|") -> torch.Tensor:
    target = torch.zeros(num_classes, dtype=torch.float32)
    parts = [x.strip() for x in str(label_str).split(sep) if x.strip() != ""]

    for p in parts:
        cls_id = int(p)

        if cls_id < 0 or cls_id >= num_classes:
            raise ValueError(f"Class id {cls_id} is out of bounds [0, {num_classes - 1}]")

        target[cls_id] = 1.0

    return target


def check_required_columns(df: pd.DataFrame, df_name: str, id_col: str, label_col: str):
    required_cols = [id_col, label_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{df_name} is missing required column: {col}")


def build_transforms(image_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


class MultiLabelImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        num_classes: int,
        transform=None,
        id_col: str = "ID",
        label_col: str = "Label",
        image_ext: str = ".png",
        label_sep: str = "|"
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.transform = transform
        self.id_col = id_col
        self.label_col = label_col
        self.image_ext = image_ext
        self.label_sep = label_sep

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_id = str(row[self.id_col])
        label_str = str(row[self.label_col])

        image_path = os.path.join(self.image_dir, image_id + self.image_ext)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = encode_multilabel(
            label_str=label_str,
            num_classes=self.num_classes,
            sep=self.label_sep
        )

        return image, target
    