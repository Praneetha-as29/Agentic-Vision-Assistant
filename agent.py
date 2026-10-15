
import os
import sys
import time
import httpx
from pathlib import Path
from google import genai
from google.genai import types
from detector import detect_objects
from rag import SafetyKnowledgeBase

MODEL = "gemini-3.8-flash"
MAX_TOOL_CALLS = 4

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
knowledge_base = None

TOOLS = [
    types.FunctionDeclaration(
        name="detect_objects",
        description="Detect objects in a local image and return labels, confidence scores, and bounding boxes.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "image_path": {"type": "STRING"}
            },
            "required": ["image_path"]
        }
    ),
    types.FunctionDeclaration(
        name="search_safety_guidelines",
        description="Search the local FAISS knowledge base for traffic safety guidelines.",
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
        if name == "detect_objects":
            return detect_objects(str(Path(arguments["image_path"]).resolve()))

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
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as error:
            if attempt == 2:
                raise
            delay = 2 ** (attempt + 1)
            print(f"Connection interrupted: {error}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as error:
            if "503" not in str(error) or attempt == 2:
                raise
            delay = 2 ** (attempt + 1)
            print(f"API temporarily unavailable. Retrying in {delay}s...")
            time.sleep(delay)

def ask_agent(image_path, question):
    image_path = str(Path(image_path).resolve())

    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                text=(
                    f"Image path: {image_path}\nQuestion: {question}\n"
                    "Use tools when helpful. Distinguish detected objects "
                    "from verified safety violations. State uncertainty "
                    "when evidence is insufficient."
                )
            )]
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a traffic-scene analysis assistant. Use tool results "
            "as evidence. Never invent detections or claim a violation "
            "solely from object labels or confidence scores."
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

    return "Stopped after reaching the agent step limit."

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python agent.py "image_path" "your question"')
        sys.exit(1)

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is missing.")
        sys.exit(1)

    try:
        print("\nFinal answer:\n", ask_agent(sys.argv[1], " ".join(sys.argv[2:])))
    except Exception as error:
        print(f"Agent failed: {type(error).__name__}: {error}")
        sys.exit(1)
