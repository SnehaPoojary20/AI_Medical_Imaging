from pathlib import Path

import torch
from torch import nn
from torchvision import models

from app.config import BACKEND_DIR


# Configure this path to your trained medical-image model weights.
MODEL_PATH = Path(BACKEND_DIR) / "models" / "densenet121_medical.pth"

# Change this to match the output classes of your trained model.
# This default is only an example for a binary classifier.
NUM_CLASSES = 2

MODEL_NAME = "DenseNet-121"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Create a DenseNet-121 architecture for the configured classes."""
    model = models.densenet121(weights=None)

    input_features = model.classifier.in_features
    model.classifier = nn.Linear(input_features, num_classes)

    return model


def load_model(
    model_path: str | Path = MODEL_PATH,
) -> nn.Module:
    """Load trained model weights and return an inference-ready model."""
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model weights were not found at: {path}. "
            "Place compatible medical-image weights at this path."
        )

    model = build_model()

    try:
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load model weights from {path}."
        ) from exc

    # Support either a raw state dictionary or a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("The model checkpoint has an invalid format.")

    # Handle checkpoints saved from torch.nn.DataParallel.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint does not match the configured DenseNet-121 "
            "architecture or number of classes. Check the model training "
            "configuration and NUM_CLASSES."
        ) from exc

    model.to(DEVICE)
    model.eval()

    return model



          

   
