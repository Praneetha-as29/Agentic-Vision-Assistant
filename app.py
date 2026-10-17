
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from agent import ask_agent
from detector import detect_objects

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Agentic Vision Assistant",
    page_icon="🔎",
    layout="wide"
)

st.title("Agentic Vision Assistant")
st.caption(
    "AI-assisted traffic-scene analysis using object detection, "
    "retrieval-augmented generation, and an AI agent."
)

if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is missing. Check your local .env file.")
    st.stop()

st.info(
    "This prototype assists with traffic-scene understanding. "
    "It does not independently verify traffic violations."
)

uploaded_file = st.file_uploader(
    "Upload a traffic-scene image",
    type=["jpg", "jpeg", "png"]
)

question = st.text_area(
    "Ask a question about the scene",
    value=(
        "Describe the visible traffic scene and identify potential "
        "safety concerns. Distinguish observations from confirmed violations. "
        "Use the supplied detection results as evidence, avoid raw bounding-box "
        "coordinates in the narrative, and acknowledge uncertainty."
    ),
    height=100
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded image")
    st.image(image, width=800)

    if st.button("Analyze traffic scene", type="primary"):
        if not question.strip():
            st.warning("Please enter a question before analyzing.")
            st.stop()

        image_path = None

        try:
            suffix = Path(uploaded_file.name).suffix.lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temporary_file:
                image.save(
                    temporary_file,
                    format="JPEG" if suffix in [".jpg", ".jpeg"] else "PNG"
                )
                image_path = temporary_file.name

            with st.spinner("Detecting objects with the Indian-road model..."):
                detections = detect_objects(image_path)

            st.subheader("Detection dashboard")

            if detections:
                counts = Counter(item["label"] for item in detections)
                columns = st.columns(min(len(counts), 4))

                for index, (label, count) in enumerate(sorted(counts.items())):
                    with columns[index % len(columns)]:
                        st.metric(label=label.replace("_", " ").title(), value=count)

                original = cv2.imread(image_path)

                if original is None:
                    raise ValueError("The uploaded image could not be read.")

                image_height, image_width = original.shape[:2]

                for item in detections:
                    x1, y1, x2, y2 = [
                        int(round(value)) for value in item["bbox"]
                    ]

                    x1 = max(0, min(x1, image_width - 1))
                    y1 = max(0, min(y1, image_height - 1))
                    x2 = max(0, min(x2, image_width - 1))
                    y2 = max(0, min(y2, image_height - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    box_width = x2 - x1
                    box_height = y2 - y1

                    cv2.rectangle(
                        original,
                        (x1, y1),
                        (x2, y2),
                        (0, 220, 100),
                        1
                    )

                    if box_width >= 55 and box_height >= 30:
                        label = (
                            f'{item["label"].replace("_", " ")} '
                            f'{item["confidence"]:.2f}'
                        )

                        font_scale = 0.38
                        thickness = 1

                        (text_width, text_height), baseline = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            thickness
                        )

                        label_y = max(y1, text_height + baseline + 2)
                        label_top = max(0, label_y - text_height - baseline - 4)
                        label_right = min(
                            image_width - 1,
                            x1 + text_width + 6
                        )

                        cv2.rectangle(
                            original,
                            (x1, label_top),
                            (label_right, label_y),
                            (0, 150, 65),
                            -1
                        )

                        cv2.putText(
                            original,
                            label,
                            (x1 + 3, label_y - baseline - 1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA
                        )

                annotated = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

                image_column, results_column = st.columns([1, 1])

                with image_column:
                    st.markdown("**Detected objects**")
                    st.image(annotated, width="stretch")

                with results_column:
                    st.markdown("**Detection details**")
                    st.dataframe(
                        [
                            {
                                "Object": item["label"].replace("_", " ").title(),
                                "Confidence": round(item["confidence"], 4),
                                "Bounding box": str(item["bbox"])
                            }
                            for item in detections
                        ],
                        width="stretch",
                        hide_index=True
                    )

                st.caption(
                    f"Showing {len(detections)} detections. Small boxes retain "
                    "their outlines; labels are hidden on very small boxes to "
                    "reduce visual clutter. All detections remain in the table."
                )

            else:
                st.warning("No objects were detected in this image.")

            with st.spinner("Retrieving guidelines and generating the assessment..."):
                answer = ask_agent(image_path, question.strip(), detections)

            st.subheader("AI safety assessment")
            st.markdown(answer)

            with st.expander("Understanding the results"):
                st.write(
                    "Object counts show the number of detections returned by "
                    "the model. Confidence scores describe the model's "
                    "confidence in each object detection, not the probability "
                    "of a traffic violation. Bounding boxes show approximate "
                    "object locations. Some labels are hidden on small boxes "
                    "to improve readability, but the detections remain in the "
                    "table. Unclear images may not provide enough evidence to "
                    "assess compliance with traffic rules."
                )

        except Exception as error:
            st.error(f"Analysis failed: {type(error).__name__}: {error}")

        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

st.divider()
st.caption(
    "Educational prototype. AI-generated assessments may be incomplete "
    "or incorrect and should not be treated as official enforcement decisions."
)
