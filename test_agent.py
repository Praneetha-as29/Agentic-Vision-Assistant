
import json
import pytest
from pathlib import Path
from unittest.mock import patch

import agent


def test_agent_requires_detection_evidence():
    with pytest.raises(ValueError, match="Detection results must be supplied"):
        agent.ask_agent("test.jpg", "Describe this scene")


def test_agent_rejects_missing_image():
    with pytest.raises(FileNotFoundError):
        agent.ask_agent(
            "image_that_does_not_exist.jpg",
            "Describe this scene",
            []
        )


def test_agent_prompt_uses_supplied_detection_evidence(tmp_path):
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"test")

    detections = [
        {
            "label": "autorickshaw",
            "confidence": 0.91,
            "bbox": [10, 20, 100, 150]
        },
        {
            "label": "person",
            "confidence": 0.88,
            "bbox": [120, 30, 160, 180]
        }
    ]

    captured = {}

    class FakeResponse:
        candidates = [type("Candidate", (), {
            "content": type("Content", (), {})()
        })()]
        function_calls = []
        text = "One autorickshaw and one person were detected."

    def fake_generate(contents, config):
        captured["contents"] = contents
        captured["config"] = config
        return FakeResponse()

    with patch.object(agent, "generate_with_retry", side_effect=fake_generate):
        answer = agent.ask_agent(
            str(image_path),
            "Describe this scene",
            detections
        )

    prompt = captured["contents"][0].parts[0].text
    assert json.dumps({"autorickshaw": 1, "person": 1}) in prompt
    assert "Detection records" in prompt
    assert "10, 20, 100, 150" not in prompt
    assert answer == "One autorickshaw and one person were detected."
