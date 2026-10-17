# Agentic Vision Assistant

An AI-powered traffic-scene analysis prototype combining YOLOv8 object detection, Retrieval-Augmented Generation (RAG), and Gemini-powered agentic tool calling.

## Overview

Agentic Vision Assistant analyzes traffic images, identifies visible objects, retrieves relevant traffic-safety guidelines, and generates an evidence-aware scene assessment.

The application provides an interactive Streamlit interface with annotated images, object counts, detection confidence scores, bounding-box coordinates, and a natural-language safety assessment.

The system is designed to distinguish detected objects from verified traffic violations and communicate uncertainty when visual evidence is insufficient.

## Key Features

- **Object Detection:** Uses pretrained YOLOv8 to identify general objects in traffic scenes.
- **Visual Annotations:** Draws bounding boxes and labels on detected objects.
- **Detection Dashboard:** Displays object counts, confidence scores, and bounding-box coordinates.
- **Retrieval-Augmented Generation:** Uses Sentence Transformers embeddings and FAISS similarity search to retrieve relevant safety guidelines.
- **Agentic Tool Calling:** Uses the Gemini API to coordinate object detection and safety-guideline retrieval.
- **Evidence-Aware Assessment:** Generates explanations that distinguish visual observations from confirmed violations.
- **Automated Testing:** Includes pytest tests for object detection and guideline retrieval.

## System Architecture

1. The user uploads a traffic image and enters a question through the Streamlit interface.
2. YOLOv8 detects visible objects and returns labels, confidence scores, and bounding-box coordinates.
3. The application displays the detections and annotated image.
4. The Gemini agent can call the object-detection and safety-guideline retrieval tools.
5. The RAG component retrieves relevant passages from the local safety-guideline documents using Sentence Transformers and FAISS.
6. The agent combines available evidence into a natural-language traffic-scene assessment.
7. The interface presents the results while highlighting evidence limitations.

## Technology Stack

- Python 3.11
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Sentence Transformers
- FAISS
- Google Gemini API
- Streamlit
- pytest

## Project Structure

```text
Agentic-Vision-Assistant/
├── agent.py
├── app.py
├── detector.py
├── rag.py
├── test_detector.py
├── test_rag.py
├── test.jpg
├── requirements.txt
├── .gitignore
├── README.md
├── docs/
│   └── safety_guidelines.txt
└── screenshots/
```

## Getting Started

### 1. Clone the repository

```powershell
git clone https://github.com/Praneetha-as29/Agentic-Vision-Assistant.git
cd Agentic-Vision-Assistant
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a file named `.env` in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your own key obtained through Google AI Studio.

**Security:** Never commit `.env` or expose your API key in screenshots, source code, or public logs. The project's `.gitignore` excludes `.env` files.

### 5. Run the web application

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, normally `http://localhost:8501`.

Upload a traffic image, enter a question, and select **Analyze traffic scene** to view the detections and generated assessment.

### 6. Run object detection independently

```powershell
python detector.py test.jpg
```

### 7. Query the local safety knowledge base

```powershell
python rag.py
```

Enter a traffic-safety question when prompted.

### 8. Run the agent from the terminal

```powershell
python agent.py test.jpg "Describe the visible traffic scene and potential safety concerns."
```

### 9. Run automated tests

```powershell
python -m pytest -v
```

The current test suite covers object-detection output validation, missing-image handling, guideline loading, empty queries, and retrieval results.

## Limitations

- The pretrained YOLOv8 model detects general object categories; it is not a dedicated traffic-violation classifier.
- Object detection alone cannot establish whether a person is wearing a compliant helmet or seatbelt.
- A single image cannot reliably establish speed, movement, right-of-way, or the full context of a traffic event.
- Language-model responses may be incomplete or incorrect.
- Gemini API access requires network connectivity and is subject to model availability and usage limits.

This project is an educational prototype and should not be used as the sole basis for real-world traffic enforcement or safety decisions.

## Future Improvements

- Evaluate performance on a broader collection of traffic images.
- Improve retrieval quality and evaluate relevance across different queries.
- Add specialized helmet and seatbelt detection models where suitable datasets are available.
- Expand automated tests and add structured evaluation metrics.
- Improve logging, API failure handling, and deployment readiness.

## Project Status

Working prototype with object detection, local safety-guideline retrieval, Gemini agentic tool calling, a Streamlit dashboard, and automated tests.

## License

No license has been selected yet. A suitable license should be added before explicitly permitting reuse or redistribution.
