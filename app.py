
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

st.set_page_config(
    page_title="Agentic Vision Assistant",
    page_icon="🔎",
    layout="wide"
)

st.title("Agentic Vision Assistant")
st.caption(
    "AI-assisted traffic-scene analysis using YOLOv8, "
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
        "safety concerns. Distinguish observations from confirmed violations."
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
                image.save(temporary_file, format="JPEG" if suffix in [".jpg", ".jpeg"] else "PNG")
                image_path = temporary_file.name

            with st.spinner("Detecting objects with YOLOv8..."):
                detections = detect_objects(image_path)

            st.subheader("Detection dashboard")

            if detections:
                counts = Counter(item["label"] for item in detections)
                columns = st.columns(min(len(counts), 4))

                for index, (label, count) in enumerate(sorted(counts.items())):
                    with columns[index % len(columns)]:
                        st.metric(label=label.title(), value=count)

                original = cv2.imread(image_path)

                for item in detections:
                    x1, y1, x2, y2 = [
                        int(value) for value in item["bbox"]
                    ]
                    label = f'{item["label"]} {item["confidence"]:.2f}'

                    cv2.rectangle(
                        original,
                        (x1, y1),
                        (x2, y2),
                        (0, 220, 100),
                        2
                    )

                    cv2.putText(
                        original,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 220, 100),
                        2,
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
                                "Object": item["label"],
                                "Confidence": round(item["confidence"], 4),
                                "Bounding box": str(item["bbox"])
                            }
                            for item in detections
                        ],
                        width="stretch",
                        hide_index=True
                    )

            else:
                st.warning("No objects were detected in this image.")

            with st.spinner("Retrieving guidelines and generating the assessment..."):
                answer = ask_agent(image_path, question.strip())

            st.subheader("AI safety assessment")
            st.markdown(answer)

            with st.expander("Understanding the results"):
                st.write(
                    "Object counts show the number of detections returned by "
                    "the model. Confidence scores describe the model's "
                    "confidence in each object detection, not the probability "
                    "of a traffic violation. Bounding boxes show approximate "
                    "object locations. Unclear images may not provide enough "
                    "evidence to assess compliance with traffic rules."
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
