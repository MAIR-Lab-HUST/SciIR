import json
import os
import time
import random
import base64
import mimetypes
import re
from openai import OpenAI
from tqdm import tqdm

# ================= Configuration =================
API_KEY = os.getenv("OPENAI_API_KEY", "")
IMAGE_FOLDER = ""  # Folder path where generated images are stored
PROMPT_FILE = "Cot_All_Three/prompt.json"
CHECKLIST_FILE = "Cot_All_Three/selected_data.json"
OUTPUT_FILE = "Cot_All_Three/evaluation_results.json"

# Retry configuration
MAX_RETRIES = 5  # Maximum number of retries
BASE_DELAY = 2  # Base delay in seconds

# Configure OpenAI client
client = OpenAI(api_key=API_KEY)


MODEL_NAME = "gemini-3-pro-preview"

# ================= System Prompt =================
SYSTEM_PROMPT_TEXT = """
Role
You are a Senior Scientific Image Reviewer. Your task is to evaluate a generated scientific figure against a specific checklist of requirements. You must be precise, hallucination-free, and strict regarding scientific accuracy.

Input Format
You will receive:
1. A Scientific Image (generated based on a prompt).
2. Original Input Prompt: The full text description used to generate the image (for context).
3. Validation Checklist (JSON) containing specific questions.

Evaluation Criteria
For each question in the checklist, perform the following steps:

1. Visual Evidence Retrieval: Look at the image to find the specific element mentioned in the question.
2. Category-Specific Logic:
   - Style: Does the overall image look like the requested format?
   - Text (Strict Text Rendering): Check Spelling and Position.
   - EntityStructure : Focus on shape, color, components.
   - ScientificProcess : Focus on arrows, lines, gradients.
   - ScientificLaw : Focus on interactions, trajectories, angles.
3. Reasoning: Formulate a brief, one-sentence justification based *only* on visual observation.
4. Verdict: Assign "Yes" (Pass) or "No" (Fail).

 Constraints
- Priority: The Checklist is your primary evaluation rubric.
- Strictness: Text errors -> Fail in "Text". Graphical errors -> Fail in Scientific categories.
- No Hallucination: Do not claim an element is present if it is not clearly visible.
- Output: You must output ONLY a valid JSON object matching the requested schema.

Output Schema
{
  "evaluation_results": [
    {
      "category": "<Category from input>",
      "answer": "Yes" | "No",
      "reason": "<Brief visual evidence focusing on graphical elements for scientific categories>"
    },
    ...
  ]
}
"""


# ================= Helpers =================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_original_prompt_text(filename, mode, prompt_map):
    item = prompt_map.get(filename)
    if not item: return None
    if mode == "cot":
        return item.get("sci-RCoT", "")
    elif mode == "abstract":
        return item.get("science_abstract_prompt", "")
    else:
        return item.get("sci-RCoT", "")


def encode_image(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None: mime_type = "image/png"
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_str, mime_type


def clean_json_text(text):
    """
    Cleanup: if the model returned a Markdown code block (```json ... ```),
    strip the markers so json.loads can parse it.
    """
    # Remove ```json and ``` markers
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def evaluate_image_with_retry(image_path, original_prompt_text, checklist_data):
    checklist_str = json.dumps(checklist_data, indent=2)
    user_text_content = f"""
[ORIGINAL INPUT PROMPT]
{original_prompt_text}
[END PROMPT]

[CHECKLIST JSON DATA]
{checklist_str}
[END DATA]

Please evaluate the provided image against every question in the checklist above.
"""

    try:
        base64_image, mime_type = encode_image(image_path)
    except Exception as e:
        return {"error": f"Failed to encode image: {str(e)}"}

    retries = 0
    while retries <= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEXT},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text_content},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high"
                        }}
                    ]}
                ],
                temperature=0,
                top_p=1,
                # response_format={"type": "json_object"} removed from create()
            )

            result_text = response.choices[0].message.content

            # Clean possible Markdown markers
            cleaned_text = clean_json_text(result_text)

            result_json = json.loads(cleaned_text)
            return result_json

        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                print(f"\n[Error] Failed to process {os.path.basename(image_path)}: {str(e)}")
                return {"error": f"Max retries reached. Error: {str(e)}"}

            sleep_time = (BASE_DELAY * (2 ** (retries - 1))) + random.uniform(0, 1)
            print(f"\n[Warning] Retry {retries}/{MAX_RETRIES} for {os.path.basename(image_path)}: {e}")
            time.sleep(sleep_time)


# ================= Main =================

def main():
    if not os.path.exists(PROMPT_FILE) or not os.path.exists(CHECKLIST_FILE):
        print("Error: Data files not found.")
        return

    prompt_data_list = load_json(PROMPT_FILE)
    checklist_data_list = load_json(CHECKLIST_FILE)
    prompt_map = {item['image_filename']: item for item in prompt_data_list}
    results = []

    print(f"Starting evaluation for {len(checklist_data_list)} items (No JSON Mode)...")

    for entry in tqdm(checklist_data_list):
        image_filename = entry['image_path']
        mode = entry['used_prompt_mode']
        checklist_obj = entry['generated_checklist']

        full_image_path = os.path.join(IMAGE_FOLDER, image_filename)
        if not os.path.exists(full_image_path):
            results.append({"image_filename": image_filename, "error": "Image file not found"})
            continue

        original_prompt_text = get_original_prompt_text(image_filename, mode, prompt_map)
        if not original_prompt_text:
            results.append({"image_filename": image_filename, "error": "Prompt text not found"})
            continue

        eval_result = evaluate_image_with_retry(full_image_path, original_prompt_text, checklist_obj)

        results.append({
            "image_filename": image_filename,
            "mode": mode,
            "evaluation": eval_result
        })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()