import base64
import os
import json
import argparse
import re
import time
from tqdm import tqdm
from google import genai
#请安装google-genai库（注意不是google)
from google.genai import types

# -----------------------------------------------------------------------------
# 1. System Instruction (定义的生成规则)
# -----------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """Role
You are an expert in evaluation design for scientific image generation benchmarks. Your task is to generate a JSON checklist for VLM scoring based on the instructions provided by the user.

Context
You need to generate checkpoints for two layers for each test sample based on the data below.
Input Data Definitions
Core Track Type: The specific track (EntityStructure, ScientificProcess, or ScientificLaw).
Input Prompt: The complete prompt containing text requirements and scientific descriptions.
Reasoning: Structured terms extracted from reference materials.
Generation Rules
Part 1: General Rules (Applies to all Prompts)
Layer 1 - Text Check (Text Rendering):
Input Processing: - ITERATE through all text explicitly required for rendering in the Input Prompt (look for quotes, "Label...", "Text..."). 
Execution:
 - FOR EACH text item found: - GENERATE a specific question checking: 1. "Spelling Correctness" 2. "Positional Accuracy" (IF AND ONLY IF the position is explicitly specified in the Prompt). 
Negative Constraints (CRITICAL):
- SCOPE: Questions must ONLY evaluate the text string itself (Spelling, Existence, Position).
- NO VISUALS: Do NOT verify visual attributes of the object the text is on (e.g., ignore color, shape, arrow direction).
- NO HALLUCINATION: Strictly based on the position described in the original text, speculation is prohibited. If there is no explicit preposition in the original Text (such as "Object labeled 'Text'"), it cannot be assumed as "inside". Always use "near" or "associated with".
Category: "Text"
Part 2: Track-Customized Rules (Core Scientific Content)
Mapping and Filtering Logic:
1、Select Term Source: Identify the correct list of terms based on the Core Track Type.
2、Intersection Verification: Identify terms that exist in BOTH the selected reasoning source AND are explicitly mentioned in the Input Prompt.
3、Attribute-Based Decomposition (One-to-Many Logic):
For EACH identified term, analyze its context in the Input Prompt to find distinct visual descriptors (adjectives, verbs, spatial constraints).
If the Prompt specifies MULTIPLE distinct visual requirements for a single term (e.g., specific shape AND specific color AND specific action), generate SEPARATE questions for each requirement.
If the Prompt only mentions the term generally, generate one comprehensive question.
4、Negative Constraint Injection (Hallucination Defense):
For each track, you must generate at least one "Negative Check" question specifically designed to catch hallucinations relevant to that scientific domain. Use the negative strategies below.
CRITICAL CONSTRAINT FOR PART 2 
NO TEXT CHECKING: Do NOT ask about labels/text. Focus ONLY on visual representation.
ATOMICITY: Each question must focus on ONE single visual attribute to ensure precise evaluation.

Question Formulation by Track:
Construct Yes/No questions focusing on specific VISUAL ATTRIBUTES based on the decomposition above:
ScientificLaw(Focus: Logic & Constraints)
Definition: Focuses on laws, principles, and constraints.
Positive Check Strategy: Decompose complex laws into specific scientific constraints.
Negative Check Strategy (Hallucination):Check for violations of fundamental domain rules (axioms). Ensure no "Impossible States" exist (e.g., objects defying gravity, inconsistent lighting/reflections, chemically impossible bonds like pentavalent carbon) and that symbolic labels match the visual logic without gibberish or data-visual contradictions.
Category: "ScientificLaw"

EntityStructure(Focus: Composition & Topology)
Definition: Focuses on scientific entities (nouns).
Positive Check Strategy: Decompose into Morphological (Shape), Chromatic (Color), and Component (Parts) or other structural checks.
Negative Check Strategy (Hallucination):Check for structural coherence. Ensure distinct objects are clearly separated (not fused) and that the entity is free of structural impossibilities.
Category: "EntityStructure"

ScientificProcess(Focus: Flow & Causality)
Definition: Focuses on flows, steps, and interactions.
Positive Strategy: Decompose into Directional (Arrows/Flow), Phase (State changes), and Interaction checks.
Negative Check Strategy (Hallucination): Check for flow logic conservation. Ensure the diagram depicts only the requested stages without hallucinated "ghost" steps, and that all directional indicators (arrows) have valid start and end points (no orphaned loops).
Category: "ScientificProcess"

Output Requirement
Format: Output pure JSON, containingchecklist (a list of question objects). Do not output markdown code blocks or explanatory text, just the raw JSON string.
Output Format Example
{
"checklist": [
{
"question": "Is the label '[Text A]' spelled correctly and positioned [Position Requirement]?",
"category": "Text"
},
{
"question": "[Question about visual features,NOT the label]?",
"category": "[ScientificLaw/EntityStructure/ScientificProcess]"
}
]
}"""


# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------

def clean_json_string(text):
    """
    清理 LLM 返回的文本，移除 Markdown 代码块标记，提取纯 JSON 字符串。
    """
    # 移除 ```json 和 ``` 标记
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()


def construct_input_text(item, mode):
    """
    根据数据项和模式构建发送给 LLM 的具体 Input 内容。
    """
    # 1. 确定 Input Prompt 来源
    if mode == 'cot':
        prompt_text = item.get("sci-RCoT", "")
    elif mode == 'abstract':
        prompt_text = item.get("science_abstract_prompt", "")
    else:
        raise ValueError("Unknown mode. Use 'cot' or 'abstract'.")

    # 2. 确定 Core Track Type (多选)
    # 逻辑：遍历 reasoning 中的 key，如果 value 不是 null，则视为该 Track 存在
    reasoning_data = item.get("reasoning", {})
    active_tracks = [k for k, v in reasoning_data.items() if v is not None]
    core_track_str = ", ".join(active_tracks) if active_tracks else "General"

    # 3. 格式化 Reasoning 内容
    reasoning_str = json.dumps(reasoning_data, indent=2)

    # 4. 组合最终 Input 文本
    input_content = f"""
Core Track Type: {core_track_str}

Input Prompt:
{prompt_text}

Reasoning:
{reasoning_str}
"""
    return input_content


def call_gemini(client, input_text, model_name="gemini-3-pro-preview"):
    """
    调用 Gemini API 并返回生成的文本。
    """
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,  # 保持低温度以获得稳定的格式
        top_p=1,
        # Thinking config as per your request
        thinking_config=types.ThinkingConfig(
            include_thoughts=False  # 我们只需要最终的 JSON
        ),
        system_instruction=[
            types.Part.from_text(text=SYSTEM_INSTRUCTION),
        ],
        response_mime_type="application/json"  # 强制模型输出 JSON
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )
        return response.text
    except Exception as e:
        print(f"\n[Error] API Call failed: {e}")
        return None


# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Checklists using Gemini.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON file containing samples.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument("--mode", "-m", type=str, choices=['cot', 'abstract'], default='cot',
                        help="Prompt source: 'cot' for sci-RCoT, 'abstract' for science_abstract_prompt.")
    parser.add_argument("--api_key", type=str, default='YOUR_GOOGLE_GENAI_API_KEY', help="Google GenAI API Key.")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: GEMINI_API_KEY not found. Set it in env or pass via --api_key.")
        return

    # 初始化 Client
    client = genai.Client(api_key=args.api_key)

    # 读取输入文件
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 结果容器
    results = []

    print(f"Starting generation for {len(data)} items in mode: {args.mode}...")

    # 遍历处理
    for item in tqdm(data):
        input_text = construct_input_text(item, args.mode)

        # 调用 API
        raw_response = call_gemini(client, input_text)

        generated_checklist = None
        if raw_response:
            try:
                clean_txt = clean_json_string(raw_response)
                generated_checklist = json.loads(clean_txt)
            except json.JSONDecodeError:
                print(f"\n[Warning] JSON Decode failed for image: {item.get('image_path')}")
                generated_checklist = {"error": "Invalid JSON returned", "raw": raw_response}

        # 构建结果对象，保留原始信息方便对照
        result_item = {
            "image_path": item.get("image_path"),
            "used_prompt_mode": args.mode,
            "core_tracks": [k for k, v in item.get("reasoning", {}).items() if v is not None],
            "generated_checklist": generated_checklist
        }
        results.append(result_item)

        # 可选：防止速率限制 (Rate Limit)，适当 sleep
        time.sleep(1)

        # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Results saved to {args.output}")


if __name__ == "__main__":
    main()