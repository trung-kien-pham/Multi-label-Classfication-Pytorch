# Multi-Label Classification

A PyTorch project for **multi-label image classification** using CSV annotations and torchvision backbones such as **ResNet**, **EfficientNet**, **ConvNeXt**, and **Swin Transformer**.

## Project Structure

```text
MULTI-LABEL_CLASSIFICATION/
├── dataset/
│   ├── images/
│   └── label_csv/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── outputs/
│   ├── best_model.pth
│   └── train_log.csv
├── data.py
├── loss.py
├── model.py
├── predict.py
└── train.py
```

## Features

- Multi-label classification from CSV annotations
- Automatic inference of number of classes from label files
- Support for multiple pretrained torchvision models:
  - `resnet18`
  - `resnet50`
  - `efficientnet_b0`
  - `efficientnet_v2_s`
  - `convnext_small`
  - `swin_t`
- Training with validation and optional test evaluation
- Save best model checkpoint
- Predict on a single image or an entire folder
- Export prediction results to CSV

## Dataset Format

Images are stored in:

```text
dataset/images/
```

CSV label files are stored in:

```text
dataset/label_csv/
```

Each CSV file should contain at least 2 columns:

- `ID`: image filename without extension or matching your configured format
- `Label`: multi-label targets separated by `|`

Example:

```csv
ID,Label
image_001,0|2
image_002,1
image_003,0|3|4
```

## Requirements

Install dependencies first:

```bash
pip install torch torchvision torchmetrics pandas numpy pillow tqdm
```

## Training

Example training command:

```bash
python train.py --train_csv "dataset/label_csv/train.csv" --val_csv "dataset/label_csv/val.csv" --test_csv "dataset/label_csv/test.csv" --image_dir "dataset/images" --output_dir "outputs" --model_name resnet50 --image_size 384 --batch_size 8 --epochs 20 --lr 1e-4 --threshold 0.4
```

### Main Arguments

- `--train_csv`: path to training CSV
- `--val_csv`: path to validation CSV
- `--test_csv`: path to test CSV
- `--image_dir`: folder containing images
- `--output_dir`: folder to save checkpoint and logs
- `--model_name`: model architecture
- `--image_size`: resize image to this size
- `--batch_size`: batch size
- `--epochs`: number of training epochs
- `--lr`: learning rate
- `--threshold`: probability threshold for multi-label prediction
- `--num_classes`: optional, auto-inferred if omitted

## Prediction

### Predict a single image

```bash
python predict.py --ckpt_path "outputs_test/best_model.pth" --image_path "dataset/images/example.jpg"
```

### Predict all images in a folder

```bash
python predict.py --ckpt_path "outputs_test/best_model.pth" --image_dir "dataset/images" --output_csv "outputs_test/predictions.csv" --print_each
```

## Output Files

After training, the output directory may contain:

- `best_model.pth`: best checkpoint based on validation Macro-F1
- `train_log.csv`: training history log

Example:

```text
outputs_test/
├── best_model.pth
└── train_log.csv
```

## Code Organization

- `train.py`: main training script
- `predict.py`: inference script for single image or folder
- `model.py`: model builder for all supported backbones
- `loss.py`: loss definition
- `data.py`: dataset, label encoding, and transforms

## Evaluation Metric

This project uses **Macro-F1** for multi-label classification evaluation.  
Macro-F1 is suitable when class imbalance exists because it gives equal importance to each class.

## Notes

- Make sure image filenames in CSV match files inside `dataset/images`
- If your CSV `ID` already contains file extensions, set arguments accordingly in the code or input format
- For prediction, the checkpoint stores:
  - model name
  - number of classes
  - threshold
  - label settings
<!-- 
## Future Improvements

- Add class name mapping for readable predictions
- Add Grad-CAM visualization
- Add weighted loss for class imbalance
- Add support for test-time augmentation -->
