import numpy as np
import torch
from PIL import Image

from app.ml.model_loader import DEVICE


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    target_class: int | None = None,
    overlay_alpha: float = 0.4,
) -> Image.Image:
    """
    Generate a Grad-CAM heatmap overlay for DenseNet-121.

    Args:
        model: A loaded DenseNet-121 model.
        input_tensor: Preprocessed image with shape [1, 3, H, W].
        original_image: Original RGB image to display beneath the heatmap.
        target_class: Output class to explain. Defaults to the top prediction.
        overlay_alpha: Heatmap opacity, between 0 and 1.

    Returns:
        An RGB PIL image containing the heatmap overlay.
    """
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be between 0 and 1.")

    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError(
            "Grad-CAM expects one image with tensor shape [1, 3, H, W]."
        )

    # DenseNet-121's final convolutional feature block.
    target_layer = model.features.denseblock4

    activations = {}

    def save_activations(module, inputs, output):
        activations["features"] = output
        output.retain_grad()

    hook = target_layer.register_forward_hook(save_activations)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        # Gradients are required, so do not use inference_mode here.
        with torch.enable_grad():
            logits = model(input_tensor.to(DEVICE))

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            if not 0 <= target_class < logits.shape[1]:
                raise ValueError("target_class is outside the model output range.")

            score = logits[0, target_class]
            score.backward()

            feature_maps = activations["features"]
            gradients = feature_maps.grad

            if gradients is None:
                raise RuntimeError(
                    "Could not calculate Grad-CAM gradients."
                )

            # Average gradients over the spatial dimensions.
            weights = gradients.mean(dim=(2, 3), keepdim=True)

            # Weighted combination of the feature maps.
            cam = (weights * feature_maps).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)

            cam = torch.nn.functional.interpolate(
                cam,
                size=original_image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

            cam = cam[0, 0].detach().cpu().numpy()

        minimum = float(cam.min())
        maximum = float(cam.max())

        if maximum > minimum:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = np.zeros_like(cam)

        # Convert normalized values into a simple red heatmap.
        red = np.clip(cam * 255, 0, 255).astype(np.uint8)
        zeros = np.zeros_like(red)
        heatmap = Image.fromarray(
            np.stack([red, zeros, zeros], axis=-1),
            mode="RGB",
        )

        base_image = original_image.convert("RGB").resize(
            heatmap.size,
            Image.Resampling.LANCZOS,
        )

        return Image.blend(base_image, heatmap, overlay_alpha)

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
