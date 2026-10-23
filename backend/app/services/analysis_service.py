from pathlib import Path
from uuid import uuid4

from app.config import BACKEND_DIR
from app.ml.inference import get_model, predict_image, prepare_tensor
from app.ml.gradcam import generate_gradcam
from app.utils.image_processing import load_image
from app.utils.validators import validate_image_upload


OUTPUT_DIR = Path(BACKEND_DIR) / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_image(filename: str, content: bytes) -> dict:
    """Validate an image, run inference, and save a Grad-CAM heatmap."""

    # 1. Validate the uploaded file.
    validate_image_upload(filename, content)

    # 2. Run model inference.
    prediction_result = predict_image(content)

    # 3. Prepare the same image as a tensor for Grad-CAM.
    original_image, input_tensor = prepare_tensor(content)

    # 4. Generate a heatmap for the predicted class.
    model = get_model()
    heatmap = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        original_image=original_image,
        target_class=prediction_result["class_index"],
    )

    # 5. Save the heatmap under a unique filename.
    heatmap_filename = f"{uuid4().hex}.png"
    heatmap_path = OUTPUT_DIR / heatmap_filename
    heatmap.save(heatmap_path, format="PNG")

    # 6. Return data for the API layer.
    return {
        "study_id": uuid4().hex,
        "model_name": prediction_result["model_name"],
        "prediction": prediction_result["prediction"],
        "confidence": prediction_result["confidence"],
        "heatmap_url": f"/api/analysis/heatmaps/{heatmap_filename}",
        "disclaimer": (
            "For educational purposes only — not for clinical use."
        ),
    }
