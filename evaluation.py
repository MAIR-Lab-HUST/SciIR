import json
import os
import time
import random
import google.generativeai as genai
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = "AIzaSyCSv8Ycw2_C0yFvRKcTU787GDHaKbZemvA"  # 填入你的 Google API Key
IMAGE_FOLDER = ""  # 生成图片存放的文件夹路径
PROMPT_FILE = "./selected_images_100/prompt.json"
CHECKLIST_FILE = "./selected_images_100/selected_data.json"
OUTPUT_FILE = "./selected_images_100/evaluation_results.json"

# 重试配置
MAX_RETRIES = 5  # 最大重试次数
BASE_DELAY = 2  # 基础等待时间（秒）

# 配置 Gemini 模型
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-1.5-pro"

generation_config = {
    "temperature": 0,
    "top_p": 1,
    "response_mime_type": "application/json",
}

# ================= 系统 Prompt 定义 =================
# 注意：移除了原本末尾的 "Here is the data... [IMAGE]"，因为这部分现在属于 System Instruction，数据由 User 输入提供
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

# ================= 模型初始化 (修改处 1) =================
# 将 System Prompt 传入 system_instruction 参数
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    system_instruction=SYSTEM_PROMPT_TEXT
)

# ================= 辅助函数 =================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_original_prompt_text(filename, mode, prompt_map):
    item = prompt_map.get(filename)
    if not item:
        return None

    if mode == "cot":
        return item.get("sci-RCoT", "")
    elif mode == "abstract":
        return item.get("science_abstract_prompt", "")
    else:
        return item.get("sci-RCoT", "")


def evaluate_image_with_retry(image_path, original_prompt_text, checklist_data):
    """
    带重试机制的评估函数
    """
    # 准备 Prompt 文本 (User Prompt)
    checklist_str = json.dumps(checklist_data, indent=2)
    user_prompt_content = f"""
[ORIGINAL INPUT PROMPT]
{original_prompt_text}
[END PROMPT]

[CHECKLIST JSON DATA]
{checklist_str}
[END DATA]

Please evaluate the provided image against every question in the checklist above.
"""

    retries = 0
    while retries <= MAX_RETRIES:
        try:
            # 1. 上传图片
            sample_file = genai.upload_file(path=image_path, display_name="Scientific Image")

            # 2. 调用模型 (修改处 2)
            # 这里不再传入 SYSTEM_PROMPT，只传入 [文件, 用户文本]
            # System Instruction 已经在模型初始化时内置了
            response = model.generate_content([sample_file, user_prompt_content])

            # 3. 尝试解析 JSON
            result_json = json.loads(response.text)

            # 成功则直接返回
            return result_json

        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                print(f"\n[Error] Failed to process {os.path.basename(image_path)} after {MAX_RETRIES} retries.")
                return {"error": f"Max retries reached. Last error: {str(e)}"}

            # 指数退避算法
            sleep_time = (BASE_DELAY * (2 ** (retries - 1))) + random.uniform(0, 1)
            print(
                f"\n[Warning] Error on {os.path.basename(image_path)}: {e}. Retrying in {sleep_time:.2f}s... ({retries}/{MAX_RETRIES})")
            time.sleep(sleep_time)


# ================= 主逻辑 =================

def main():
    print("Loading data...")
    try:
        prompt_data_list = load_json(PROMPT_FILE)
        checklist_data_list = load_json(CHECKLIST_FILE)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file. {e}")
        return

    prompt_map = {item['image_filename']: item for item in prompt_data_list}
    results = []

    print(f"Starting evaluation for {len(checklist_data_list)} items with Retry Mode (System Instruction Separated)...")

    # 使用 tqdm 显示进度条
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

        # 调用带重试的评估函数
        eval_result = evaluate_image_with_retry(full_image_path, original_prompt_text, checklist_obj)

        results.append({
            "image_filename": image_filename,
            "mode": mode,
            "evaluation": eval_result
        })

    # 保存最终结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()