<div align="center">

# 🧠 SegmentAI — Neural Image Segmentation

**Pixel-level road scene segmentation powered by UNet-ResNet34 and PyTorch**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[Try Demo](#-quick-start) · [Architecture](#-architecture) · [Training](#-training-pipeline) · [Evaluation](#-evaluation)

</div>

---

## 📋 About the Project

SegmentAI is a deep learning project for **binary image segmentation** of road scenes. It takes a road image as input and predicts a pixel-level mask that separates the road surface from the background.

The project includes:
- 🔬 **UNet with ResNet34 encoder** — pretrained on ImageNet for strong feature extraction
- 📊 **Complete training pipeline** — with data augmentation, hybrid loss, and LR scheduling
- 🌐 **Flask web application** — drag-and-drop interface for real-time inference
- 📈 **Evaluation suite** — Dice, IoU, Precision, Recall, and Pixel Accuracy metrics

### How It Works

```
Road Image (RGB)  →  UNet-ResNet34  →  Binary Mask  →  Overlay Visualization
     ↓                                      ↓
  256×256 input                     Road vs Background
```

The model was trained on **290 paired road images** with corresponding ground truth masks. It uses a combination of **CrossEntropy + Dice loss** for balanced pixel-level optimization.

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        UNet Architecture                           │
│                                                                    │
│   Input (3×256×256)                        Output (2×256×256)      │
│        │                                         ▲                 │
│        ▼                                         │                 │
│   ┌─────────┐     Skip Connections      ┌──────────────┐           │
│   │ Encoder │ ──────────────────────►   │   Decoder    │           │
│   │(ResNet34)│                           │  (UpConv +   │          │
│   │         │                           │   Concat)    │           │
│   │ Block 1 │ ─────────────────────►    │  UpBlock 4   │           │
│   │  64 ch  │                           │   64 ch      │           │
│   │         │                           │              │           │
│   │ Block 2 │ ─────────────────────►    │  UpBlock 3   │           │
│   │  128 ch │                           │   128 ch     │           │
│   │         │                           │              │           │
│   │ Block 3 │ ─────────────────────►    │  UpBlock 2   │           │
│   │  256 ch │                           │   256 ch     │           │
│   │         │                           │              │           │
│   │ Block 4 │ ─────────────────────►    │  UpBlock 1   │           │
│   │  512 ch │                           │   512 ch     │           │
│   └────┬────┘                           └──────────────┘           │
│        │                                         ▲                 │
│        └──────► Bottleneck (512 ch) ─────────────┘                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Details |
|-----------|---------|
| **Encoder** | ResNet34 pretrained on ImageNet |
| **Decoder** | Transposed convolutions with skip connections |
| **Loss Function** | 0.5 × CrossEntropy + 0.5 × Dice Loss |
| **Optimizer** | Adam (lr=1e-4, weight_decay=1e-5) |
| **Scheduler** | ReduceLROnPlateau (patience=3) |
| **Input Size** | 256 × 256 RGB |
| **Output** | 2-class probability map (road / background) |
| **Augmentation** | Flip, Rotate, ColorJitter, GaussianBlur |

---

## 📁 Project Structure

```
SegmentAI/
├── app/
│   ├── server.py              # Flask web server
│   ├── model.py               # Model inference wrapper
│   ├── templates/
│   │   └── index.html         # Web UI
│   └── static/
│       ├── css/style.css      # Grey-themed stylesheet
│       └── js/app.js          # Frontend logic
├── pipeline/
│   ├── config.py              # Hyperparameters & paths
│   ├── dataset.py             # Dataset class & augmentations
│   ├── train.py               # Training script (CLI)
│   └── evaluate.py            # Metrics evaluation
├── data/
│   ├── Training_Images/       # 290 road scene images
│   └── Ground_Truth/          # Corresponding binary masks
├── weights/                   # Model weights (.pth files)
├── outputs/                   # Evaluation results
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended) or CPU
- ~500 MB disk space for model weights

### 1. Clone the Repository

```bash
git clone https://github.com/SaiSriKumar6686/Image-Segmentation.git
cd Image-Segmentation
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Data & Weights

Place your files in the correct directories:

```
data/
├── Training_Images/    ← Put road images here (1.jpg, 2.jpg, ...)
└── Ground_Truth/       ← Put mask images here (1.png, 2.png, ...)

weights/
├── unet_resnet34_road_segmentation.pth     ← Trained model weights
└── best_unet_checkpoint.pth                ← Best checkpoint
```

> **Note:** Model weights are not included in the repo due to size (~390 MB). Train the model using the pipeline or contact the author for pre-trained weights.

### 5. Launch the Web App

```bash
cd app
python server.py
```

Open your browser at **http://localhost:5000** — upload any road image and click **Run Segmentation**.

---

## 🏋️ Training Pipeline

Train the model from scratch or resume from a checkpoint:

```bash
# Train with default settings (25 epochs, batch_size=8, lr=1e-4)
cd app
python -m pipeline.train

# Custom training
python -m pipeline.train --epochs 50 --batch-size 4 --lr 0.0005

# Resume from checkpoint
python -m pipeline.train --resume --epochs 30
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 25 | Number of training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--resume` | False | Resume from last checkpoint |

### What Happens During Training

1. Dataset is loaded and split (90% train / 10% validation)
2. Images are augmented with flips, rotations, color jitter, and blur
3. Model trains with combined CrossEntropy + Dice loss
4. Best checkpoint is saved when validation Dice score improves
5. Learning rate reduces automatically when Dice plateaus
6. Final model weights are saved to `weights/`

---

## 📈 Evaluation

Run comprehensive evaluation on the validation set:

```bash
cd app
python -m pipeline.evaluate
```

This computes and saves the following metrics to `outputs/evaluation_metrics.json`:

| Metric | Description |
|--------|-------------|
| **Dice Coefficient** | Overlap between predicted and true mask (F1 for segmentation) |
| **IoU (Jaccard)** | Intersection over Union of predicted and true regions |
| **Precision** | Fraction of predicted positive pixels that are correct |
| **Recall** | Fraction of actual positive pixels that are detected |
| **Pixel Accuracy** | Overall percentage of correctly classified pixels |

---

## 🌐 Web Application

The Flask web app provides a browser-based interface for real-time segmentation:

- **Drag & Drop** — Drop any road image onto the upload zone
- **Instant Results** — See original, mask, and overlay side-by-side
- **Metrics Display** — Confidence, coverage, and pixel counts
- **Responsive Design** — Works on desktop and mobile

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web UI |
| `POST` | `/api/segment` | Accepts image upload, returns segmentation |
| `GET` | `/api/health` | Server and model health check |

### API Usage Example

```bash
curl -X POST -F "image=@road.jpg" http://localhost:5000/api/segment
```

Response:
```json
{
    "success": true,
    "mask": "<base64_png>",
    "overlay": "<base64_png>",
    "confidence": 78.5,
    "coverage": 34.2,
    "segmented_pixels": 22413,
    "total_pixels": 65536
}
```

---

## 🛠 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **PyTorch** | Deep learning framework |
| **Segmentation Models PyTorch** | UNet architecture |
| **Flask** | Web server & API |
| **OpenCV** | Image processing |
| **Albumentations** | Data augmentation |
| **Pillow** | Image encoding |
| **NumPy** | Numerical operations |
| **tqdm** | Progress bars |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built with 🤍 by [Sai Sri Kumar](https://github.com/SaiSriKumar6686)**

</div>
