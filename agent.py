
import os
import sys
import time
import json
import httpx
from pathlib import Path
from google import genai
from google.genai import types
from rag import SafetyKnowledgeBase
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
MAX_TOOL_CALLS = 3

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
knowledge_base = None

TOOLS = [
    types.FunctionDeclaration(
        name="search_safety_guidelines",
        description="Search the local FAISS knowledge base for relevant traffic safety guidelines.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING"}
            },
            "required": ["query"]
        }
    )
]

def run_tool(name, arguments):
    global knowledge_base

    try:
        if name == "search_safety_guidelines":
            if knowledge_base is None:
                knowledge_base = SafetyKnowledgeBase()
            return knowledge_base.search(arguments["query"])

        return {"error": "Unknown tool"}

    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}

def generate_with_retry(contents, config):
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config
            )
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout):
            if attempt == 2:
                raise
            delay = 2 ** (attempt + 1)
            print(f"Connection interrupted. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as error:
            if "503" not in str(error) or attempt == 2:
                raise
            delay = 2 ** (attempt + 1)
            print(f"API temporarily unavailable. Retrying in {delay}s...")
            time.sleep(delay)

def ask_agent(image_path, question, detections=None):
    image_path = str(Path(image_path).resolve())

    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if detections is None:
        raise ValueError("Detection results must be supplied to ground the assessment.")

    evidence = [
        {
            "label": item["label"],
            "confidence": item["confidence"]
        }
        for item in detections
    ]

    counts = {}
    for item in evidence:
        counts[item["label"]] = counts.get(item["label"], 0) + 1

    user_prompt = (
        f"Question: {question}\n\n"
        "Use the following object-detection results from the uploaded image "
        "as the source of object labels and counts. Do not run another object "
        "detector or invent additional objects.\n\n"
        f"Detection counts: {json.dumps(counts)}\n"
        f"Detection records (labels and confidence only): {json.dumps(evidence)}\n\n"
        "Describe visible objects using these results. Treat detections as "
        "model predictions, not guaranteed ground truth. Overlapping or "
        "incorrect detections may occur. Do not mention raw bounding-box "
        "coordinates. Discuss general safety concerns only when supported "
        "by the available evidence. A single static image cannot establish "
        "speeding or other motion-dependent violations. Clearly distinguish "
        "observations, potential concerns, and confirmed violations."
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)]
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=(
            "You are an evidence-aware traffic-scene analysis assistant. "
            "The supplied detection results are the only permitted source "
            "for object labels and counts. Do not invent objects, counts, "
            "road markings, signals, or vehicle behavior. Use the safety "
            "guideline search tool when relevant. Distinguish observations "
            "from potential risks and confirmed violations. Never claim a "
            "violation solely from object labels or confidence scores. "
            "Write a concise, natural-language answer with short headings "
            "where useful. Do not reveal raw coordinates or internal details."
        ),
        tools=[types.Tool(function_declarations=TOOLS)]
    )

    tool_calls = 0

    for _ in range(MAX_TOOL_CALLS + 1):
        response = generate_with_retry(contents, config)

        if not response.candidates:
            return "The model returned no response."

        contents.append(response.candidates[0].content)
        calls = response.function_calls or []

        if not calls:
            return response.text or "No text answer was returned."

        results = []

        for call in calls:
            if tool_calls >= MAX_TOOL_CALLS:
                result = {"error": "Tool-call limit reached"}
            else:
                print(f"Calling tool: {call.name}")
                result = run_tool(call.name, dict(call.args or {}))
                tool_calls += 1

            results.append(types.Part.from_function_response(
                name=call.name,
                response={"result": result}
            ))

        contents.append(types.Content(role="user", parts=results))

    return "The analysis stopped after reaching the tool-call limit."

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python agent.py "image_path" "your question"')
        sys.exit(1)

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is missing.")
        sys.exit(1)

    try:
        from detector import detect_objects
        detections = detect_objects(sys.argv[1])
        print(
            "\nFinal answer:\n",
            ask_agent(sys.argv[1], " ".join(sys.argv[2:]), detections)
        )
    except Exception as error:
        print(f"Agent failed: {type(error).__name__}: {error}")
        sys.exit(1)
