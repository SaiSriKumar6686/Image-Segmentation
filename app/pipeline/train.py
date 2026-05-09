import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import segmentation_models_pytorch as smp
from tqdm import tqdm

from pipeline.config import (
    DEVICE, TRAIN_IMG_DIR, TRAIN_MASK_DIR, MODEL_PATH, CHECKPOINT_PATH,
    BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EPOCHS, NUM_CLASSES, LEARNING_RATE,
    WEIGHT_DECAY, TRAIN_SPLIT
)
from pipeline.dataset import (
    SegmentationDataset, get_train_transforms, get_eval_transforms, safe_collate
)


def dice_loss(pred, target, smooth=1e-6):
    pred_prob = torch.softmax(pred, dim=1)[:, 1]
    target_float = (target == 1).float()
    inter = (pred_prob * target_float).sum()
    return 1 - (2.0 * inter + smooth) / (pred_prob.sum() + target_float.sum() + smooth)


def dice_score(pred, target, smooth=1e-6):
    pred_bin = (torch.softmax(pred, dim=1)[:, 1] > 0.5).float()
    target_f = (target == 1).float()
    inter = (pred_bin * target_f).sum()
    return (2.0 * inter + smooth) / (pred_bin.sum() + target_f.sum() + smooth)


def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, count = 0.0, 0
    for batch in tqdm(loader, desc="  Training", leave=False):
        if batch is None:
            continue
        images, masks = batch
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = 0.5 * criterion(outputs, masks) + 0.5 * dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        count += images.size(0)
    return total_loss / count if count else 0.0


@torch.no_grad()
def validate(model, loader):
    model.eval()
    scores = []
    for batch in tqdm(loader, desc="  Validating", leave=False):
        if batch is None:
            continue
        images, masks = batch
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True).long()
        scores.append(dice_score(model(images), masks).item())
    return np.mean(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train UNet segmentation model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    # Datasets
    train_ds = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, get_train_transforms())
    eval_ds = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, get_eval_transforms())
    print(f"Dataset: {len(train_ds)} image-mask pairs")

    n = len(train_ds)
    t_size = int(TRAIN_SPLIT * n)
    indices = list(range(n))
    train_idx, val_idx = indices[:t_size], indices[t_size:]

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, collate_fn=safe_collate, pin_memory=True, num_workers=0)
    val_loader = DataLoader(Subset(eval_ds, val_idx), batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, collate_fn=safe_collate, pin_memory=True, num_workers=0)

    # Model, optimizer, scheduler
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_dice = 0, 0.0
    if args.resume and CHECKPOINT_PATH and __import__("os").path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_dice = ckpt.get("dice_score", 0.0)
        print(f"Resumed from epoch {start_epoch}, best dice {best_dice:.4f}")

    # Training loop
    print(f"\n{'='*50}")
    print(f"Training for {args.epochs - start_epoch} epochs (batch_size={args.batch_size}, lr={args.lr})")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_dice = validate(model, val_loader)
        scheduler.step(val_dice)
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:>3}/{args.epochs}  |  loss: {loss:.4f}  |  dice: {val_dice:.4f}  |  "
              f"lr: {lr_now:.1e}  |  {elapsed:.1f}s")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dice_score": val_dice
            }, CHECKPOINT_PATH)
            print(f"  ✓ Saved checkpoint (dice={val_dice:.4f})")

    # Save final weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nTraining complete. Best dice: {best_dice:.4f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
