
import pytest
from detector import detect_objects


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
