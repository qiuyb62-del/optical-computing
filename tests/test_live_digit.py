import numpy as np
import pytest

from lumerical.live_digit import preprocess_canvas


def test_preprocess_rejects_blank_canvas():
    blank = np.full((280, 280), 255.0)
    with pytest.raises(ValueError, match="empty canvas"):
        preprocess_canvas(blank.ravel(), 280, 280)


def test_preprocess_centers_a_stroke_in_28x28():
    canvas = np.full((280, 280), 255.0)
    canvas[40:240, 130:155] = 0.0
    image, meta = preprocess_canvas(canvas.ravel(), 280, 280)
    assert image.shape == (28, 28)
    assert image.dtype == np.uint8
    assert image.max() == 255
    assert meta["source_width"] == 280
    assert meta["bbox"] == [130, 40, 155, 240]
    # centered: the stroke should occupy the middle rows/columns
    assert image[10:18, 13:15].min() == 255
    assert image[:3].max() == 0
    assert image[-3:].max() == 0


def test_preprocess_rejects_wrong_pixel_count():
    with pytest.raises(ValueError, match="pixel count"):
        preprocess_canvas([0.0] * 10, 28, 28)
