import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .config import NORM_MEAN, NORM_STD, IMAGE_SIZE


class SegmentationDataset(Dataset):
    """Road segmentation dataset with automatic image-mask pairing."""

    MASK_EXTENSIONS = [".png", "_mask.png", ".gif", "_mask.gif"]

    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.pairs = self._discover_pairs()

    def _discover_pairs(self):
        images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        pairs = []
        for img in images:
            base = os.path.splitext(img)[0]
            for ext in self.MASK_EXTENSIONS:
                mask_name = f"{base}{ext}"
                if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                    pairs.append((img, mask_name))
                    break
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        image = cv2.imread(os.path.join(self.image_dir, img_name))
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.int64)

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return image, mask


def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2()
    ], is_check_shapes=False)


def get_eval_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2()
    ], is_check_shapes=False)


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
