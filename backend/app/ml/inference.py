from functools import lru_cache

import torch
from PIL import Image
from torchvision import transforms

from app.ml.model_loader import (
    DEVICE,
    MODEL_NAME,
    load_model,
)
from app.utils.image_processing import preprocess_image


IMAGE_SIZE = (224, 224)

# These are neutral placeholder labels, not disease diagnoses.
# Replace them with the exact class names used during training.
CLASS_LABELS = ["class_0", "class_1"]


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@lru_cache(maxsize=1)
def get_model():
    """Load the model once and reuse it for subsequent requests."""
    return load_model()


def prepare_tensor(content: bytes) -> tuple[Image.Image, torch.Tensor]:
    """Decode an image and convert it into a model input tensor."""
    image = preprocess_image(
        content,
        target_size=IMAGE_SIZE,
    )

    tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    return image, tensor.to(DEVICE)


def predict_image(content: bytes) -> dict:
    """Run inference and return the predicted class and confidence."""
    model = get_model()
    _, input_tensor = prepare_tensor(content)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    if len(CLASS_LABELS) != probabilities.numel():
        raise RuntimeError(
            "CLASS_LABELS does not match the model output size. "
            "Configure the labels used during training."
        )

    class_index = int(torch.argmax(probabilities).item())

    return {
        "model_name": MODEL_NAME,
        "prediction": CLASS_LABELS[class_index],
        "class_index": class_index,
        "confidence": float(probabilities[class_index].item()),
        "probabilities": {
            label: float(probabilities[index].item())
            for index, label in enumerate(CLASS_LABELS)
        },
    }
