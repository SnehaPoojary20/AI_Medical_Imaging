import pytest

from app.utils.image_utils import validate_image_extension


def test_valid_png_extension():
    assert validate_image_extension("scan.png") is True


def test_valid_jpg_extension():
    assert validate_image_extension("scan.jpg") is True


def test_valid_jpeg_extension():
    assert validate_image_extension("scan.jpeg") is True


def test_invalid_extension():
    assert validate_image_extension("document.pdf") is False


def test_extension_check_is_case_insensitive():
    assert validate_image_extension("scan.PNG") is True