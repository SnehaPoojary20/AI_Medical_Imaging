
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


def load_image(content: bytes) -> Image.Image:
    """Load image bytes and return an RGB image."""
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()

            # Correct orientation using image metadata, if present.
            image = ImageOps.exif_transpose(image)

            # Convert to a consistent color format.
            return image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(
            "Unable to load the image. "
            "Please provide a valid supported image."
        ) from exc


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """Resize an image to (width, height)."""
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("Image dimensions must be positive.")

    return image.resize(
        size,
        Image.Resampling.LANCZOS,
    )


def preprocess_image(
    content: bytes,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Load an image and optionally resize it for model input."""
    image = load_image(content)

    if target_size is not None:
        image = resize_image(image, target_size)

    return image

#Image decoding and preparation