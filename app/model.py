import io
import base64
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationModel:
    """Wrapper for loading and running inference with the UNet model."""

    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    IMG_SIZE = 256
    NUM_CLASSES = 2

    def __init__(self, weights_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=self.NUM_CLASSES
        ).to(self.device)
        self._load_weights(weights_path)
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(self.IMG_SIZE, self.IMG_SIZE),
            A.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD),
            ToTensorV2()
        ])

    def _load_weights(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)

    @torch.no_grad()
    def predict(self, image_bytes):
        """Run segmentation on raw image bytes. Returns (mask_b64, overlay_b64, confidence)."""
        # Decode image
        img_array = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError("Could not decode image")
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        h, w = original.shape[:2]

        # Preprocess
        transformed = self.transform(image=original_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Inference
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        mask = (probs > 0.5).astype(np.uint8)
        confidence = float(probs.mean())

        # Resize mask back to original dimensions
        mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Create colored mask (cyan tint for segmented regions)
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colored_mask[mask_full == 1] = [0, 220, 220]

        # Overlay on original
        overlay = cv2.addWeighted(original_rgb, 0.6, colored_mask, 0.4, 0)

        # Binary mask visualization (white on black)
        mask_vis = (mask_full * 255).astype(np.uint8)

        return {
            "mask": self._to_base64(mask_vis, is_gray=True),
            "overlay": self._to_base64(overlay),
            "confidence": round(confidence * 100, 2),
            "segmented_pixels": int(mask_full.sum()),
            "total_pixels": int(h * w),
            "coverage": round(float(mask_full.sum()) / (h * w) * 100, 2)
        }

    @staticmethod
    def _to_base64(img_array, is_gray=False):
        if is_gray:
            pil_img = Image.fromarray(img_array, mode="L")
        else:
            pil_img = Image.fromarray(img_array, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
