import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import segmentation_models_pytorch as smp

from pipeline.config import (
    DEVICE, TRAIN_IMG_DIR, TRAIN_MASK_DIR, MODEL_PATH,
    NUM_CLASSES, EVAL_BATCH_SIZE, TRAIN_SPLIT, OUTPUTS_DIR
)
from pipeline.dataset import SegmentationDataset, get_eval_transforms, safe_collate

import os


def compute_metrics(pred, target):
    """Compute Dice, IoU, Precision, Recall, Pixel Accuracy for a batch."""
    pred_bin = (torch.softmax(pred, dim=1)[:, 1] > 0.5).float()
    target_f = (target == 1).float()
    smooth = 1e-6

    tp = (pred_bin * target_f).sum()
    fp = (pred_bin * (1 - target_f)).sum()
    fn = ((1 - pred_bin) * target_f).sum()

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    total_pixels = target_f.numel()
    correct = ((pred_bin == target_f).float()).sum()
    pixel_acc = correct / total_pixels

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "pixel_accuracy": pixel_acc.item()
    }


@torch.no_grad()
def evaluate():
    print(f"Device: {DEVICE}")

    # Load model
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=NUM_CLASSES).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    print("Model loaded successfully")

    # Dataset (validation split only)
    dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, get_eval_transforms())
    n = len(dataset)
    val_start = int(TRAIN_SPLIT * n)
    val_indices = list(range(val_start, n))
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, collate_fn=safe_collate, num_workers=0)

    print(f"Evaluating on {len(val_indices)} samples...\n")

    all_metrics = []
    for batch in tqdm(val_loader, desc="Evaluating"):
        if batch is None:
            continue
        images, masks = batch
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True).long()
        outputs = model(images)
        all_metrics.append(compute_metrics(outputs, masks))

    if not all_metrics:
        print("No valid batches found.")
        return

    # Aggregate
    keys = all_metrics[0].keys()
    summary = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}

    print(f"\n{'='*40}")
    print(f"  Evaluation Results ({len(all_metrics)} samples)")
    print(f"{'='*40}")
    for k, v in summary.items():
        print(f"  {k:<20s}: {v:.4f}")
    print(f"{'='*40}")

    # Save to JSON
    out_path = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    evaluate()
