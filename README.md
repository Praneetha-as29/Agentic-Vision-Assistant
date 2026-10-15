# Agentic Vision Assistant

An AI-powered traffic-scene analysis prototype combining YOLOv8 object detection, Retrieval-Augmented Generation (RAG), and Gemini-powered agentic tool calling.

## Overview

The Agentic Vision Assistant analyzes traffic images using a computer vision detector and a local traffic-safety knowledge base. A Gemini-powered agent can call tools to detect objects and retrieve relevant safety guidelines before generating an evidence-aware response.

The goal is to distinguish visible objects from confirmed traffic violations and communicate uncertainty when the available evidence is insufficient.

## Features

- YOLOv8 object detection with labels, confidence scores, and bounding boxes.
- Sentence Transformers embeddings for traffic-safety guideline retrieval.
- FAISS similarity search over local text documents.
- Gemini API function calling to coordinate available tools.
- Basic exception handling and a configurable tool-call limit.
- Evidence-aware responses that avoid treating object detections as proof of violations.

## Architecture

1. The user supplies an image and a question.
2. The Gemini agent decides which tools to call.
3. YOLOv8 detects objects in the image.
4. The RAG component retrieves relevant safety guidelines from local documents.
5. The agent combines available results into a traffic-scene assessment.

## Technology Stack

- Python
- Ultralytics YOLOv8
- OpenCV
- Sentence Transformers
- FAISS
- Google Gemini API
- NumPy

## Project Structure

```text
Agentic Vision Assistant/
|-- agent.py
|-- detector.py
|-- rag.py
|-- docs/
|   |-- safety_guidelines.txt
|-- test.jpg
|-- requirements.txt
|-- .gitignore
|-- README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Agentic Vision Assistant"
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Set your key in PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

Get an API key through Google AI Studio. Never commit your real API key to the repository.

### 5. Run object detection

```powershell
python detector.py test.jpg
```

### 6. Query the safety knowledge base

```powershell
python rag.py
```

Enter a traffic-safety question when prompted.

### 7. Run the agent

```powershell
python agent.py test.jpg "What objects are visible, and what safety concerns can be assessed?"
```

The first run may download model weights and embedding models. The agent requires network access to the Gemini API. Availability and free-tier quotas may vary.

## Limitations

- The pretrained YOLOv8 model detects general objects; it is not a dedicated traffic-violation classifier.
- Detecting a person, motorcycle, or car does not establish a helmet or seatbelt violation.
- A single image cannot reliably establish speed, movement, right-of-way, or the full context of a traffic event.
- Results depend on image quality, object detections, retrieved guidelines, and language-model output.
- Gemini is required for the agentic reasoning workflow; object detection and guideline retrieval can be run separately.

This prototype should not be used as the sole basis for real-world enforcement or safety decisions.

## Future Improvements

- Add an image-upload interface.
- Improve retrieval by indexing individual guideline passages.
- Add automated tests and evaluate multiple traffic images.
- Explore specialized helmet and seatbelt detection models.
- Improve logging and resilience to API failures.

## Project Status

Working prototype. Object detection, local safety-guideline retrieval, and Gemini tool calling have been demonstrated. Broader testing and usability improvements remain future work.

## License

No license has been selected yet. Add an appropriate license before permitting reuse of this project.
