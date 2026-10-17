
import pytest
from pathlib import Path
from detector import detect_objects, MODEL_PATH


def test_missing_image_raises_error():
    with pytest.raises(FileNotFoundError):
        detect_objects("image_that_does_not_exist.jpg")


def test_detection_results_have_expected_fields():
    detections = detect_objects("test.jpg")

    assert isinstance(detections, list)

    for detection in detections:
        assert "label" in detection
        assert "confidence" in detection
        assert "bbox" in detection

        assert isinstance(detection["label"], str)
        assert isinstance(detection["confidence"], float)
        assert len(detection["bbox"]) == 4


def test_configured_model_file_exists():
    assert Path(MODEL_PATH).is_file()


def test_detection_confidence_is_in_valid_range():
    detections = detect_objects("test.jpg")

    for detection in detections:
        assert 0.0 <= detection["confidence"] <= 1.0


def test_bounding_box_coordinates_are_numeric():
    detections = detect_objects("test.jpg")

    for detection in detections:
        assert all(
            isinstance(value, (int, float))
            for value in detection["bbox"]
        )
