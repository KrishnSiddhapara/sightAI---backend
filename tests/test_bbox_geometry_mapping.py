"""
Unit tests for Crop-to-Full Coordinate Mapping, Tile NMS, and Background Crop Sanity Check.
"""

import pytest
from utils.coordinate_utils import (
    map_crop_box_to_full,
    apply_tile_nms,
    is_empty_background_crop,
    sanitize_box,
    calculate_iou
)
from PIL import Image, ImageDraw


def test_map_crop_box_to_full_center():
    """Tests mapping a center crop box back to full image 0-1000 coordinates."""
    # Suppose crop is located at [ymin=200, xmin=200, ymax=600, xmax=600] in full image 0-1000 scale
    crop_bounds_1000 = {"x_min": 200.0, "y_min": 200.0, "x_max": 600.0, "y_max": 600.0}
    # Center box inside crop: [xmin=250, ymin=250, xmax=750, ymax=750] (0-1000 crop relative)
    crop_box = {"x_min": 250.0, "y_min": 250.0, "x_max": 750.0, "y_max": 750.0}

    mapped = map_crop_box_to_full(crop_box, crop_bounds_1000)
    assert mapped is not None
    # Expected: x_min = 200 + 0.25*(400) = 300
    # Expected: x_max = 200 + 0.75*(400) = 500
    assert mapped["x_min"] == 300.0
    assert mapped["y_min"] == 300.0
    assert mapped["x_max"] == 500.0
    assert mapped["y_max"] == 500.0


def test_map_crop_box_to_full_corners():
    """Tests mapping corner boxes from sub-crops back to full image."""
    crop_bounds_1000 = {"x_min": 500.0, "y_min": 500.0, "x_max": 1000.0, "y_max": 1000.0}
    crop_box = {"x_min": 0.0, "y_min": 0.0, "x_max": 1000.0, "y_max": 1000.0}

    mapped = map_crop_box_to_full(crop_box, crop_bounds_1000)
    assert mapped is not None
    assert mapped["x_min"] == 500.0
    assert mapped["y_min"] == 500.0
    assert mapped["x_max"] == 1000.0
    assert mapped["y_max"] == 1000.0


def test_apply_tile_nms_deduplication():
    """Tests that overlapping tile detections (IoU >= 0.5) are deduplicated."""
    detections = [
        {"label": "elephant figurine", "box": {"x_min": 100.0, "y_min": 100.0, "x_max": 200.0, "y_max": 200.0}},
        {"label": "elephant figurine", "box": {"x_min": 105.0, "y_min": 105.0, "x_max": 205.0, "y_max": 205.0}},  # High IoU ~0.85
        {"label": "elephant figurine", "box": {"x_min": 500.0, "y_min": 500.0, "x_max": 600.0, "y_max": 600.0}},  # Distinct
    ]

    nms_result = apply_tile_nms(detections, iou_threshold=0.50)
    assert len(nms_result) == 2
    labels = [d["label"] for d in nms_result]
    assert labels.count("elephant figurine") == 2


def test_is_empty_background_crop():
    """Tests empty background crop detector on flat vs textured images."""
    # Flat image (single solid color)
    flat_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    assert is_empty_background_crop(flat_img, stddev_threshold=6.0) is True

    # High variance / textured image
    textured_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    draw = ImageDraw.Draw(textured_img)
    draw.rectangle([20, 20, 80, 80], fill=(255, 255, 255))
    assert is_empty_background_crop(textured_img, stddev_threshold=6.0) is False
