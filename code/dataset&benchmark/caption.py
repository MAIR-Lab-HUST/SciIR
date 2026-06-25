import re
import json
import base64
import os
import time
import threading
import signal
import atexit
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

API_KEY = ""


BASE_URL = ""

API_A_KEY = API_KEY
API_A_BASE = BASE_URL

API_A_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

API_B_KEY = API_KEY
API_B_BASE = BASE_URL

API_B_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


CLASSIFIED_INPUT = "7/metadata_7.json"
OUTPUT_PATH = "batch7/caption_7.json"
CACHE_PATH = "batch7/caption_cache7.json"

IMAGE_DIR = "batch7/images_7"


MAX_WORKERS = 70
MAX_RETRIES = 6
RETRY_DELAY = 2


BATCH_SIZE = 100

client_A = OpenAI(api_key=API_A_KEY, base_url=API_A_BASE)
client_B = OpenAI(api_key=API_B_KEY, base_url=API_B_BASE)



class ResultManager:
    def __init__(self, save_path, batch_size=50):
        self.save_path = save_path
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.unsaved_count = 0

        if os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                    if "processed" not in self.data: self.data["processed"] = []
                    if "outputs" not in self.data: self.data["outputs"] = []
            except json.JSONDecodeError:
                print(f"[Warn] Cache file {save_path} is corrupted; resetting.")
                self.data = {"processed": [], "outputs": []}
        else:
            self.data = {"processed": [], "outputs": []}

        self.processed_set = set(self.data["processed"])

    def is_processed(self, key):
        return key in self.processed_set

    def add_result(self, key, result_entry):
        with self.lock:
            self.data["processed"].append(key)
            self.data["outputs"].append(result_entry)
            self.processed_set.add(key)
            self.unsaved_count += 1

            if self.unsaved_count >= self.batch_size:
                self._save_to_disk_unsafe()
                self.unsaved_count = 0

    def force_save(self):
        with self.lock:
            if self.unsaved_count > 0:
                print(f"\n[System] Force-saving remaining {self.unsaved_count} items...")
                self._save_to_disk_unsafe()
                self.unsaved_count = 0
            else:
                if not os.path.exists(self.save_path):
                    self._save_to_disk_unsafe()

    def _save_to_disk_unsafe(self):
        temp_path = self.save_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.save_path)
        except Exception as e:
            print(f"[Error] Failed to save file: {e}")



def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_model(client, model, system_prompt, user_prompt, temperature=0.1, top_p=1.0, image_base64=None):
    messages = [{"role": "system", "content": system_prompt}]

    if image_base64:

        content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            }
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_prompt})

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            wait_time = RETRY_DELAY * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                print(f"[Retry {attempt + 1}] Error: {e}")
                time.sleep(wait_time)
            else:
                print(f"[Error] API Failed after max retries: {e}")

    return ""


def clean_json_output(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None



def generate_reasoning(data):
    system_prompt = """You are a scientific image reasoning annotator. Your duty is: based on the input paper text and image, extract only the scientific information explicitly supported by the image and caption, and output strict structured reasoning JSON. Do not fabricate conclusions beyond what the text and image support; do not introduce elements invisible in the image or unsupported by the caption; do not perform common sense completion. You may internally parse the input and implicitly understand the overall layout, but the final output must only contain the specified JSON fields.

Detailed Requirements:
Internally parse the input first, analyzing the overall image layout and main visual elements, prioritizing information sources: Image > Figure Title > Caption > Article Body. Note: This step is only for internal reasoning; do not list this description in the final output.

Complete the "Extracted Terms" and "Visual Description" for the corresponding information ONLY based on the selected tags in "Reasoning Ability":

ScientificLaw: Extract terms: Laws, principles, boundary conditions, applicable premises, etc. (extractable directly or implied from the graph and caption). Visualization: Manifestation of each constraint in the graph (geometric layout, labels, symbols, measurement elements, etc.).

EntityStructure: Extract terms: Scientific entity names in the image (nominalized, no verbs). Visualization: Visual characteristics of each entity (morphology, structure, color/material/texture, spatial position and scale, quantity, etc.).

ScientificProcess: Extract terms: Process names in the image (e.g., preparation process, experimental steps, time series, mechanism chain). Visualization: Visual manifestation of different stages, change conditions (e.g., arrows, timeline, loops) of this process.

Output Requirements:
Strictly follow the JSON structure below, output no extra free text.
terms: Write only terms/nouns (no sentences, punctuation, or modifiers), at the granularity confirmed by the caption and image.
visualization: For each item in terms, there must be a visual description; use concrete, restorable visual elements; do not introduce external information.
Non-empty validation: In the output JSON, "terms" and "visualization" for all selected reasoning abilities must NOT be empty lists "[]".
Unselected ability keys must be null.

Output Format:
{
  "reasoning": {
    "ScientificLaw": {
      "terms": [],
      "visualization": []
    },
    "EntityStructure": {
      "terms": [],
      "visualization": []
    },
    "ScientificProcess": {
      "terms": [],
      "visualization": []
    }
  }
}"""

    image_base64 = encode_image(data["segments"]["path"])
    user_prompt = f"""
Input:
- text: {{"article_title": "{data['article_title']}", "article_abstract": "{data['article_abstract']}", "article_body": "{data['article_body']}", "figure_title": "{data['figure_title']}"}}
- figure_caption: {{"figure_caption": "{data['figure_caption']}"}}
- reasoning_ability: {data["segments"]["labels"]}
- subject: {data["subjects"]}
"""
    result = call_model(client_A, API_A_MODEL, system_prompt, user_prompt, temperature=0.1, image_base64=image_base64)
    reasoning_json = clean_json_output(result)

    if reasoning_json is None:
        reasoning_json = {"reasoning": {}}

    if "reasoning" in reasoning_json and isinstance(reasoning_json["reasoning"], dict):
        r_data = reasoning_json["reasoning"]
        target_keys = ["ScientificLaw", "EntityStructure", "ScientificProcess"]
        for key in target_keys:
            if key in r_data:
                val = r_data[key]
                if isinstance(val, dict):
                    is_terms_empty = not val.get("terms")
                    is_viz_empty = not val.get("visualization")
                    if is_terms_empty and is_viz_empty:
                        r_data[key] = None

    return reasoning_json


def generate_cot(reasoning_json, data):
    system_prompt = """You are a scientific image visualization narrator. Your duty is: input reasoning, 
primarily based on the visualization items in reasoning, refer to the input image, 
and output a coherent, detailed scene description sci-RCoT.

Detailed Requirements:
1. Complete Visual Style:
   - Observe the original image, identify its specific drawing style 
     (e.g., Schematic diagram, Photorealistic render, etc.).
   - You must explicitly specify this style at the beginning of the generated instruction.

2. Complete Text Rendering:
   - Observe key text in the original image (labels, legends, axis titles).
   - You must include mandatory text rendering requirements in the instruction, 
     using phrases like "explicitly labeled as...", "including the text...", 
     "with axis labeled..." etc.

3. Integrate Scientific Logic:
   - Use the visualization items in reasoning to describe entity structure, 
     topological relationships, and dynamic processes.
   - Language must be coherent, building a complete scene, not a simple list.

Output Requirements:
Please do not output text directly, but output a JSON object containing the 
following two fields:
- "sci_RCoT": The generated complete visual description text.
- "rendered_text": A string list (List[str]), extracting all specific text 
  content you explicitly requested to render in sci_RCoT (such as label names, 
  axis titles, legend text, etc.).

Output Format (Must be strict JSON):
{
    "sci_RCoT": "Your detailed narrative...",
    "rendered_text": ["text_content_1", "text_content_2"]
}

The sci-RCoT must completely cover every point in the visualization array of 
each selected label in reasoning, without omission or altering its meaning, 
and strictly forbid adding any elements not appearing in reasoning, caption, 
or image; the language style should be concrete, coherent, and capable of 
restoring the scene from text."""
    image_base64 = encode_image(data["segments"]["path"])
    user_prompt = f"""
Input:
- figure_caption: {{"figure_title": "{data['figure_title']}", "figure_caption": "{data['figure_caption']}"}}
- reasoning: {json.dumps(reasoning_json, ensure_ascii=False)}
"""
    result = call_model(client_A, API_A_MODEL, system_prompt, user_prompt, temperature=0.3, image_base64=image_base64)
    result_json = clean_json_output(result)

    if result_json is None:
        return "", []
    return result_json.get("sci_RCoT", ""), result_json.get("rendered_text", [])


def generate_prompt(reasoning_json, sci_rcot, rendered_text):
    system_prompt = """Role: Scientific Image Reasoning Generation Prompt Assistant Objective: Your goal is to generate a concise, semantically compressed abstract_prompt in JSON format. You will achieve this by synthesizing input sci-RCoT with specific Reasoning dimension terms.
Input Data:
sci-RCoT: TA detailed image description composed of all the 'visualization' elements from 'reasoning'.
Reasoning: Contains specific 'terms' and their corresponding 'visualization'.
rendered_text: A list of text strings allowed to be rendered in the image.
Processing Logic:
Analyze: Read the sci-RCoT to understand the scientific semantics.
Preserve Style: Extract the visualization style requirement (typically the first sentence or phrase of sci-RCoT, e.g., "A realistic 3D render...", "A schematic diagram of...", "A cross-section view..."). This must be the opening of your abstract_prompt.
Map & Replace: Identify the description in sci-RCoT that corresponds to 'visualization' in Reasoning, and strictly replace it with the 'terms' provided in Reasoning.
Text Selection: Determine necessary text labels based on the sci-RCoT context.Include text rendering requests in abstract_prompt if they are necessary for scientific clarity or context.
Compress: Synthesize the result into an abstract_prompt without visual descriptions.
Synchronization: Extract exactly the text strings that are explicitly requested to be rendered in your generated abstract_prompt and populate the retained_text list.
Constraints & Guardrails:
- Semantic Integrity: The replacement must perfectly match the original scientific semantics.
- Style Consistency: The output must start with the original visualization style found in sci-RCoT.
- Output Format: Return only a JSON object with the following structure:
JSON
{
  "abstract_prompt": "Your concise, term-based prompt here... (e.g., 'Diagram showing [Term A] labeled 'Text1'...')",
  "retained_text": ["Text1", "Text2"] 
}
"""
    user_prompt = json.dumps({
        "reasoning": reasoning_json.get("reasoning", {}),
        "sci-RCoT": sci_rcot,
        "rendered_text_candidates": rendered_text
    }, ensure_ascii=False, indent=2)

    result = call_model(client_B, API_B_MODEL, system_prompt, user_prompt, temperature=0.3, image_base64=None)
    prompt_json = clean_json_output(result)

    if prompt_json is None:
        return "", []

    final_prompt = prompt_json.get("abstract_prompt", "")
    retained_text_list = prompt_json.get("retained_text", [])
    return final_prompt, retained_text_list




def process_single_task(task_data, manager: ResultManager):
    seg = task_data["segments"]
    key = seg.get("path") or seg.get("filename")

    if not os.path.exists(seg["path"]):
        return f"Skip: {key} (File not found at {seg['path']})"

    try:
        reasoning = generate_reasoning(task_data)
        sci_rcot, rendered_text_list = generate_cot(reasoning, task_data)
        abstract_prompt, retained_text_list = generate_prompt(reasoning, sci_rcot, rendered_text_list)

        result_entry = {
            "image_path": seg["path"],
            "reasoning": reasoning.get("reasoning", {}),
            "sci-RCoT": sci_rcot,
            "rendered_text_stage2": rendered_text_list,
            "science_abstract_prompt": abstract_prompt,
            "retained_text_stage3": retained_text_list
        }

        manager.add_result(key, result_entry)

        return f"Success: {key}"

    except Exception as e:
        return f"Failed: {key} ({str(e)})"



manager = ResultManager(CACHE_PATH, batch_size=BATCH_SIZE)


def signal_handler(sig, frame):
    print("\n[System] Interrupt signal received; saving data. Please do not force close...")
    manager.force_save()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
atexit.register(manager.force_save)

if __name__ == "__main__":
    print(f"Initializing... Image directory: {IMAGE_DIR}")
    print(f"Using API endpoint: {BASE_URL}")
    print(f"Cache path: {CACHE_PATH}, batch write size: {BATCH_SIZE}")

    classified_path = Path(CLASSIFIED_INPUT)
    if not classified_path.exists():
        print(f"Input file not found: {CLASSIFIED_INPUT}")
        exit(1)

    raw_items = json.load(open(classified_path, "r", encoding="utf-8"))

    tasks = []
    for item in raw_items:
        segs = item.get("segments", [])
        if isinstance(segs, dict): segs = [segs]
        if not segs:
            lp = item.get("local_path") or item.get("image_path")
            if lp:
                segs = [{"path": lp, "filename": Path(lp).name, "labels": []}]
            else:
                continue

        for seg in segs:
            filename = seg.get("filename")
            if not filename:
                raw_path = seg.get("path")
                if raw_path:
                    filename = os.path.basename(raw_path)

            if filename:
                new_full_path = os.path.join(IMAGE_DIR, filename)
                seg["path"] = new_full_path
                seg["filename"] = filename
            else:
                pass

            task = item.copy()
            task["segments"] = seg
            tasks.append(task)

    pending_tasks = []
    for idx, data in enumerate(tasks):
        seg = data["segments"]
        key = seg.get("path") or seg.get("filename") or str(idx)
        if not manager.is_processed(key):
            pending_tasks.append(data)

    total = len(tasks)
    processed = len(manager.data["processed"])
    print(f"Total tasks: {total}, completed: {processed}, pending: {len(pending_tasks)}")

    if pending_tasks:
        print(f"Starting processing, workers: {MAX_WORKERS}...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_key = {
                executor.submit(process_single_task, task, manager): (task["segments"].get("path") or "unknown")
                for task in pending_tasks
            }

            for future in tqdm(as_completed(future_to_key), total=len(pending_tasks), desc="Processing"):
                key = future_to_key[future]
                try:
                    msg = future.result()
                    if msg.startswith("Failed"):
                        print(f"\n{msg}")
                except Exception as exc:
                    print(f"\n[Fatal Error in Thread] {key}: {exc}")

    print(f"All tasks finished. Final results will be saved to: {OUTPUT_PATH}")
    manager.force_save()

    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(manager.data["outputs"], f, ensure_ascii=False, indent=2)
        print("Final results exported.")
    except Exception as e:
        print(f"Failed to export final results: {e}")