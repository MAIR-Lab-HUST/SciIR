import json
import os
import time
import random
import base64
import mimetypes
import re
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = "sk-JfUt2XysrnHF6RZATjog7Vhsey9rQm797LD2n1Vn8MbvimO7"
IMAGE_FOLDER = ""  # 生成图片存放的文件夹路径
PROMPT_FILE = "Cot_All_Three/prompt.json"
CHECKLIST_FILE = "Cot_All_Three/selected_data.json"
OUTPUT_FILE = "Cot_All_Three/evaluation_results.json"

# 重试配置
MAX_RETRIES = 5  # 最大重试次数
BASE_DELAY = 2  # 基础等待时间（秒）

# 配置 OpenAI 客户端
client = OpenAI(api_key=API_KEY)


MODEL_NAME = "gemini-3-pro-preview"

# ================= 系统 Prompt 定义 =================
SYSTEM_PROMPT_TEXT = """
### Role
You are a Senior Scientific Image Reviewer. Your task is to evaluate a generated scientific figure against a specific checklist of requirements. You must be precise, hallucination-free, and strict regarding scientific accuracy.

### Input Format
You will receive:
1. A **Scientific Image** (generated based on a prompt).
2. **Original Input Prompt**: The full text description used to generate the image (for context).
3. **Validation Checklist** (JSON) containing specific questions.

### Evaluation Criteria
For each question in the checklist, perform the following steps:

1. **Visual Evidence Retrieval**: Look at the image to find the specific element mentioned in the question.
2. **Category-Specific Logic**:
   - **Style**: Does the overall image look like the requested format?
   - **Text** (Strict Text Rendering): Check Spelling and Position.
   - **EntityStructure** (Visual Object Verification): Focus on shape, color, components. IGNORE text labels.
   - **ScientificProcess** (Visual Flow Verification): Focus on arrows, lines, gradients. IGNORE text labels.
   - **ScientificLaw** (Logic Verification): Focus on interactions, trajectories, angles.
3. **Reasoning**: Formulate a brief, one-sentence justification based *only* on visual observation.
4. **Verdict**: Assign "Yes" (Pass) or "No" (Fail).

### Constraints
- **Priority**: The **Checklist** is your primary evaluation rubric. 
- **Strictness**: Text errors -> Fail in "Text". Graphical errors -> Fail in Scientific categories.
- **No Hallucination**: Do not claim an element is present if it is not clearly visible.
- **Output**: You must output **ONLY** a valid JSON object matching the requested schema.

### Output Schema
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


# ================= 辅助函数 =================

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
    清理函数：如果模型返回了 Markdown 代码块 (```json ... ```)，
    这里将其去除，以便 json.loads 能正常解析。
    """
    # 移除 ```json 和 ``` 标记
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
                # create() 中已移除了 response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content

            # 清理可能存在的 Markdown 标记
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


# ================= 主逻辑 =================

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