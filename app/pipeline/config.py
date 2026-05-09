import os
import torch
import torch.backends.cudnn as cudnn

# Auto-detect project root (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "Training_Images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "Ground_Truth")

# Model paths
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "unet_resnet34_road_segmentation.pth")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "best_unet_checkpoint.pth")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Output paths
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 1
NUM_EPOCHS = 25
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
IMAGE_SIZE = 256
TRAIN_SPLIT = 0.9

# Normalization (ImageNet stats)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    cudnn.enabled = True
