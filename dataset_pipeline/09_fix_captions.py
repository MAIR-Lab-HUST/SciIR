import json
import os
import time
import threading
import signal
import atexit
import sys
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

# 尝试导入 tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ============ 配置区域 ============
API_A_KEY = "YOUR_API_KEY_HERE"
API_A_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_A_MODEL = "qwen3-vl-plus"

API_B_KEY = "YOUR_API_KEY_HERE"
API_B_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_B_MODEL = "qwen3-max"

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABSTRACTS_PATH = os.path.join(BASE_DIR, "send", "classified_metadata_1a.json")
CAPTIONS_PATH = os.path.join(BASE_DIR, "send", "caption_1a_dedup.json")
IMAGE_DIR = os.path.join(BASE_DIR, "send", "filtered_images_1a")
OUTPUT_PATH = os.path.join(BASE_DIR, "send", "caption_1a_fixed.json")
CACHE_PATH = os.path.join(BASE_DIR, "send", "caption_1a_fixed_cache.json")

# 并发配置
MAX_WORKERS = 10
MAX_RETRIES = 6
RETRY_DELAY = 2
BATCH_SIZE = 20

client_A = OpenAI(api_key=API_A_KEY, base_url=API_A_BASE)
client_B = OpenAI(api_key=API_B_KEY, base_url=API_B_BASE)

# ============ 数据管理类 (来自 pipeline_final.py) ============

class ResultManager:
    def __init__(self, save_path, batch_size=50):
        self.save_path = save_path
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.unsaved_count = 0

        # 加载现有数据
        if os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                    if "processed" not in self.data: self.data["processed"] = []
                    if "outputs" not in self.data: self.data["outputs"] = []
            except json.JSONDecodeError:
                print(f"[Warn] 缓存文件 {save_path} 损坏，将重置。")
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
                print(f"\n[System] 正在强制保存剩余的 {self.unsaved_count} 条数据...")
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
            print(f"[Error] 保存文件失败: {e}")

# ============ 工具函数 ============

def normalize_path(p):
    if not p:
        return ""
    return os.path.basename(p)

def load_json(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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

def call_model(client, model, system_prompt, user_prompt, temperature=0.1, top_p=1.0, image_base64=None):
    messages = [{"role": "system", "content": system_prompt}]
    if image_base64:
        content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
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
                time.sleep(wait_time)
            else:
                print(f"[Error] API Failed: {e}")
    return ""

# ============ 核心 Prompt 与逻辑 (来自 pipeline_final.py) ============

def generate_reasoning(data):
    system_prompt = """你是一名科学图像推理标注员。你的职责是：基于输入的论文文本与图片，只提取图像与图注明确支持的科学信息，输出严格结构化的 reasoning JSON。禁止编造超出文本与图像所支持的结论；不得引入图像中不可见或图注未支持的元素；不得进行常识性补全。内部可以先思考并对图像整体布局做隐式理解，但最终输出必须仅包含指定的 JSON 字段。
详细要求：
1) 内部先解析输入，解析图片整体布局与主要视觉元素，优先依据信息源的权重：图片>figure_title > 图注 > 文章正文。注意：该步骤仅用于内部推理，最终输出不单列此描述。
2) 仅根据“推理能力”中被选中的标签，必须完成对应信息的“提取术语”与“可视化描述”：
   - ScientificLaw（科学规律）
     提取术语：定律、原理、边界条件、适用前提等科学约束（提取能从图与图注直接支持或隐含的）。
     可视化：每项约束在图中的体现（几何布局、标注、符号、测量元素等）。
   - EntityStructure（科学实体）
     提取术语：图片中的科学实体名称（名词化，不含动词）。
     可视化：每个实体的形态结构、颜色/材质/纹理、空间位置与尺度、数量等视觉特性。
   - ScientificProcess（科学过程）
     提取术语：图片中的过程名称（如制备流程、实验步骤、时间序列、机制链条）。
     可视化：这个过程不同阶段的状态、变化条件（如箭头、时间轴、循环）的视觉体现。
3) 输出要求：
- 请用英文回答。
- 严格遵循下方 JSON 结构，不能输出任何额外自由文本。
- terms：仅写术语/名词（不含句子、标点与修饰），按图注与图像确证的粒度。
- visualization：对于terms中的每一项都要有一段可视化描述；使用具体、可还原画面的视觉要素描述；不得引入图外信息。
- 非空校验：输出的 JSON 中，所有选中的推理能力对应的 "terms" 和 "visualization" 严禁输出空列表 "[]"。
输出格式：
{{
  "reasoning": {{
    "ScientificLaw": {{"terms": [], "visualization": []}},
    "EntityStructure": {{"terms": [], "visualization": []}},
    "ScientificProcess": {{"terms": [], "visualization": []}}
  }}
}}"""

    image_base64 = encode_image(data["segments"]["path"])
    user_prompt = f"""
输入：
- text：{{"article_title": "{data['article_title']}", "article_abstract": "{data['article_abstract']}", "article_body": "{data.get('article_body', '')}", "figure_title": "{data['figure_title']}"}}
- figure_caption：{{"figure_caption": "{data['figure_caption']}"}}
- reasoning_ability：{data["segments"]["labels"]}
- subject：{data.get("subjects", [])}
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
    system_prompt = """你是一名科学图像可视化复述员。你的职责是：输入reasoning，以reasoning中的 visualization 条目为主，参考输入的图片，输出连贯、细节充分的场景化描述sci-RCoT。
详细要求：
1. 补全视觉风格：
   - 观察原图，识别其具体的绘图风格（例如：Schematic diagram, Photorealistic render等）。
   - 在生成的指令开头，必须明确指定这种风格。
2. 补全文本渲染：
   - 观察原图中的关键文字（标签、图例、轴标题）。
   - 在指令中必须包含强制性的文本渲染要求，使用 "explicitly labeled as...", "including the text...", "with axis labeled..." 等句式。
3. 整合科学逻辑：
   - 使用 reasoning 中的 visualization 条目，描述实体结构、拓扑关系和动态过程。
   - 语言必须连贯，构建一个完整的场景，而不是简单的列表。
输出要求：
请不要直接输出文本，而是输出一个 JSON 对象，包含以下两个字段：
- "sci_RCoT": 生成的完整可视化描述文本。
- "rendered_text": 一个字符串列表（List[str]），提取你在 sci_RCoT 中明确要求渲染的所有具体文本内容（如标签名、轴标题、图例文字等）。
输出格式（必须是严格的 JSON）：
{{
    "sci_RCoT": "Your detailed narrative...",
    "rendered_text": ["text_content_1", "text_content_2"]
}}
sci-RCoT要完整覆盖reasoning 中各已选标签的 visualization 数组里每一条要点，不得遗漏、不得改写其含义，严禁新增任何reasoning、图注或图像中未出现的要素；语言风格要具象、连贯、可据文还原画面。
"""
    image_base64 = encode_image(data["segments"]["path"])
    user_prompt = f"""
输入：
- figure_caption：{{"figure_title": "{data['figure_title']}", "figure_caption": "{data['figure_caption']}"}}
- reasoning：{json.dumps(reasoning_json, ensure_ascii=False)}
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
Note for "retained_text": This list must contains only strings for which explicit rendering instructions are specified in abstract_prompt,EXCLUDING strings that are merely mentioned.
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
    key = normalize_path(seg.get("path")) # 使用文件名作为 Key

    if not os.path.exists(seg["path"]):
        return f"Skip: {key} (File not found at {seg['path']})"

    try:
        # === Pipeline ===
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

        # === 写入缓存 ===
        manager.add_result(key, result_entry)
        return f"Success: {key}"

    except Exception as e:
        return f"Failed: {key} ({str(e)})"

# ============ 清理与更新逻辑 ============

def cleanup_inconsistent_data(manager, abstracts_data, captions_data):
    print("\n[System] Starting final consistency check and cleanup...")
    
    # 1. Merge fixed results into captions_data
    fixed_results = {normalize_path(item["image_path"]): item for item in manager.data["outputs"]}
    caption_map = {normalize_path(item['image_path']): item for item in captions_data}
    
    merged_count = 0
    for filename, fixed_item in fixed_results.items():
        if filename in caption_map:
            original = caption_map[filename]
            # Update fields
            original["reasoning"] = fixed_item.get("reasoning", original.get("reasoning"))
            original["sci-RCoT"] = fixed_item.get("sci-RCoT", original.get("sci-RCoT"))
            original["science_abstract_prompt"] = fixed_item.get("science_abstract_prompt", original.get("science_abstract_prompt"))
            
            # Optional: Map specific keys if needed, preserving existing ones
            if "rendered_text_stage2" in fixed_item:
                original["rendered_text"] = fixed_item["rendered_text_stage2"]
            
            merged_count += 1
    
    print(f"Merged {merged_count} fixed items into memory.")

    # 2. Identify items to delete (Check consistency again)
    items_to_delete = set()
    
    # Re-scan all abstracts to find inconsistencies
    for article in abstracts_data:
        for segment in article.get("segments", []):
            img_filename = segment.get("filename")
            if not img_filename:
                 raw_path = segment.get("path")
                 if raw_path:
                     img_filename = os.path.basename(raw_path)
            
            if not img_filename:
                continue
                
            normalized_filename = img_filename
            caption_item = caption_map.get(normalized_filename)
            
            should_delete = False
            
            if not caption_item:
                should_delete = True
            else:
                seg_labels = segment.get("labels", [])
                if seg_labels:
                    reasoning = caption_item.get("reasoning", {})
                    for label in seg_labels:
                        label_data = reasoning.get(label)
                        is_valid = False
                        if label_data and isinstance(label_data, dict):
                            terms = label_data.get("terms", [])
                            visualization = label_data.get("visualization", [])
                            if terms and len(terms) > 0 and visualization and len(visualization) > 0:
                                is_valid = True
                        
                        if not is_valid:
                            should_delete = True
                            break
            
            if should_delete:
                items_to_delete.add(normalized_filename)

    if not items_to_delete:
        print("No remaining inconsistencies found.")
    else:
        print(f"Found {len(items_to_delete)} items still inconsistent. Deleting...")
        
        # A. Delete Images
        deleted_imgs = 0
        for fname in items_to_delete:
            fpath = os.path.join(IMAGE_DIR, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    deleted_imgs += 1
                except Exception as e:
                    print(f"Failed to delete {fpath}: {e}")
        print(f"Deleted {deleted_imgs} image files.")

        # B. Update Abstracts (Remove segments)
        for article in abstracts_data:
            new_segments = []
            for segment in article.get("segments", []):
                fname = segment.get("filename") or os.path.basename(segment.get("path", ""))
                if fname not in items_to_delete:
                    new_segments.append(segment)
            article["segments"] = new_segments
        
        # Filter out articles with no segments left? 
        # User said delete records. If article has empty segments, it's effectively empty record of images.
        # But article text remains. Keeping article structure usually safer unless explicitly told to remove empty articles.
        # I will keep article but empty segments list.

        # C. Update Captions
        captions_data[:] = [item for item in captions_data if normalize_path(item.get("image_path")) not in items_to_delete]

    # 3. Save updated files
    if merged_count > 0 or items_to_delete:
        print("Saving updated datasets to source files...")
        try:
            with open(ABSTRACTS_PATH, "w", encoding="utf-8") as f:
                json.dump(abstracts_data, f, ensure_ascii=False, indent=2)
            with open(CAPTIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(captions_data, f, ensure_ascii=False, indent=2)
            print("Source files updated successfully.")
        except Exception as e:
            print(f"Error saving source files: {e}")


# ============ 主程序 ============

manager = ResultManager(CACHE_PATH, batch_size=BATCH_SIZE)

def signal_handler(sig, frame):
    print("\n[System] 捕获中断信号，正在保存数据，请勿强制关闭...")
    manager.force_save()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
atexit.register(manager.force_save)

def main():
    print("Checking inconsistencies and preparing tasks...")
    
    abstracts_data = load_json(ABSTRACTS_PATH)
    captions_data = load_json(CAPTIONS_PATH)

    if not abstracts_data or not captions_data:
        print("Data load failed. Exiting.")
        return

    # 构建 Caption 索引
    caption_map = {normalize_path(item['image_path']): item for item in captions_data}

    tasks = []
    
    for article in abstracts_data:
        for segment in article.get("segments", []):
            img_filename = segment.get("filename")
            if not img_filename:
                 raw_path = segment.get("path")
                 if raw_path:
                     img_filename = os.path.basename(raw_path)
            
            if not img_filename:
                continue

            # 规范化用于查找
            normalized_filename = img_filename
            
            # 检查是否存在于 Caption 数据中
            caption_item = caption_map.get(normalized_filename)
            
            # 构造完整的图片路径
            full_image_path = os.path.join(IMAGE_DIR, normalized_filename)

            # 如果原本就没有 Caption，或者检查出不一致，都添加到重做列表
            needs_fix = False
            
            if not caption_item:
                # print(f"Missing caption for {normalized_filename}")
                # 如果连 caption 都没有，可能原本就被过滤了，或者需要补充。
                # 这里主要关注“不一致”，即有 label 但没内容。如果原本没 caption，假设不需要处理？
                # 根据用户描述“有很多labels中标有scientificLaw的但是caption中为空”，说明是有 caption 条目的，只是内容空。
                pass 
            else:
                seg_labels = segment.get("labels", [])
                if not seg_labels:
                    continue

                reasoning = caption_item.get("reasoning", {})
                
                # 检查每个 label
                for label in seg_labels:
                    label_data = reasoning.get(label)
                    
                    is_valid = False
                    if label_data and isinstance(label_data, dict):
                        terms = label_data.get("terms", [])
                        visualization = label_data.get("visualization", [])
                        if terms and len(terms) > 0 and visualization and len(visualization) > 0:
                            is_valid = True
                    
                    if not is_valid:
                        needs_fix = True
                        break
            
            if needs_fix:
                # 检查是否已经处理过 (cache)
                if manager.is_processed(normalized_filename):
                    continue
                
                # 构造任务
                task = article.copy()
                # 确保 segment 包含正确的本地路径
                seg_copy = segment.copy()
                seg_copy["path"] = full_image_path
                seg_copy["filename"] = normalized_filename
                task["segments"] = seg_copy
                
                tasks.append(task)

    print(f"Found {len(tasks)} inconsistent items to re-process.")
    
    if not tasks:
        print("No tasks to process.")
        return

    # --- 并发执行 ---
    print(f"Starting processing with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_key = {
            executor.submit(process_single_task, task, manager): task["segments"]["filename"]
            for task in tasks
        }

        for future in tqdm(as_completed(future_to_key), total=len(tasks), desc="Fixing"):
            key = future_to_key[future]
            try:
                msg = future.result()
                if msg.startswith("Failed"):
                    print(f"\n{msg}")
            except Exception as exc:
                print(f"\n[Fatal Error in Thread] {key}: {exc}")

    print(f"All tasks completed. Saving results to: {OUTPUT_PATH}")
    manager.force_save()

    # 导出最终结果 (合并新的结果到旧的或者另存为)
    # 这里我们另存为 fix 文件，后续用户可以决定如何合并
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(manager.data["outputs"], f, ensure_ascii=False, indent=2)
        print("Fixed captions exported.")
    except Exception as e:
        print(f"Export failed: {e}")

    # 执行最终清理与更新
    cleanup_inconsistent_data(manager, abstracts_data, captions_data)

if __name__ == "__main__":
    main()

