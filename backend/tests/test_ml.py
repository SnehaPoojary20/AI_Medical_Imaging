import torch

from app.ml.inference import CLASS_LABELS, predict_image


class FakeModel(torch.nn.Module):
    """A lightweight model for testing inference."""

    def forward(self, image_tensor):
        batch_size = image_tensor.shape[0]

        # Two output scores for the two classes.
        logits = torch.tensor([[2.0, 1.0]])

        return logits.repeat(batch_size, 1)


def test_class_labels_are_defined():
    assert len(CLASS_LABELS) == 2
    assert all(isinstance(label, str) for label in CLASS_LABELS)


def test_prediction_returns_valid_class(monkeypatch):
    fake_model = FakeModel()

    # Replace the real model loader with our lightweight model.
    monkeypatch.setattr(
        "app.ml.inference.get_model",
        lambda: fake_model,
    )

    # Replace image preprocessing with a dummy tensor.
    monkeypatch.setattr(
        "app.ml.inference.prepare_tensor",
        lambda image: torch.zeros((1, 3, 224, 224)),
    )

    result = predict_image("dummy_image.png")

    assert result is not None
    assert result["class_index"] in range(len(CLASS_LABELS))
    assert result["prediction"] in CLASS_LABELS
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == len(CLASS_LABELS)