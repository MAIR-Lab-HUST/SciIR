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

# 尝试导入 tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ============ 配置区域 ============
API_A_KEY = "sk-a6e5442b7edf46e2a1d39351875309de"
API_A_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_A_MODEL = "qwen3-vl-plus"

API_B_KEY = "sk-a6e5442b7edf46e2a1d39351875309de"
API_B_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_B_MODEL = "qwen3-max"

# 输入路径 (用于找回原始 Prompt 上下文)
CLASSIFIED_INPUT = "scir_dataset/classified_abstracts3.json"
# 待修复的文件 (既是输入也是输出)
OUTPUT_PATH = "scir_dataset/caption3.json"
# 图片文件夹路径
IMAGE_DIR = "scir_dataset/filtered_images_3"

# 并发配置
MAX_WORKERS = 15
MAX_RETRIES = 6  # API 网络层面的重试
RETRY_DELAY = 2
BATCH_SIZE = 20  # 每修复多少条保存一次

# 逻辑重试配置 (新增)
MAX_LOGIC_RETRIES = 2  # 尝试生成的次数，失败则删除

client_A = OpenAI(api_key=API_A_KEY, base_url=API_A_BASE)
client_B = OpenAI(api_key=API_B_KEY, base_url=API_B_BASE)


# ============ 数据修复管理类 ============

class RepairManager:
    def __init__(self, target_file, batch_size=20):
        self.target_file = target_file
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.unsaved_count = 0

        # 加载待修复的数据集
        if os.path.exists(target_file):
            with open(target_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, dict):
                    self.data_list = content.get("outputs", [])
                else:
                    self.data_list = content
        else:
            print(f"[Error] 找不到文件: {target_file}")
            sys.exit(1)

        # 建立索引： image_path -> list_index
        self.index_map = {item["image_path"]: i for i, item in enumerate(self.data_list)}

    def update_entry(self, index, new_entry):
        """线程安全地更新列表中的特定条目"""
        with self.lock:
            self.data_list[index] = new_entry
            self.unsaved_count += 1
            if self.unsaved_count >= self.batch_size:
                self._save_to_disk_unsafe()
                self.unsaved_count = 0

    def delete_entry_and_file(self, index, image_path):
        """线程安全地标记删除条目并删除物理文件"""
        with self.lock:
            # 1. 内存中标记为 None (不直接 pop，防止影响其他线程的索引)
            self.data_list[index] = None

            # 2. 删除物理文件
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print(f"[Delete] 已删除图片文件: {image_path}")
                except Exception as e:
                    print(f"[Error] 删除文件失败 {image_path}: {e}")

            self.unsaved_count += 1
            if self.unsaved_count >= self.batch_size:
                self._save_to_disk_unsafe()
                self.unsaved_count = 0

    def force_save(self):
        """强制保存"""
        with self.lock:
            self._save_to_disk_unsafe()
            self.unsaved_count = 0

    def _save_to_disk_unsafe(self):
        temp_path = self.target_file + ".tmp"
        try:
            # 关键修改：保存时过滤掉 None 的条目，实现真正的记录删除
            valid_data = [item for item in self.data_list if item is not None]

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(valid_data, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.target_file)
            print(f"[System] 已保存进度 (有效数据: {len(valid_data)} 条)。")
        except Exception as e:
            print(f"[Error] 保存失败: {e}")


# ============ 校验逻辑 ============

def is_entry_complete(entry):
    """
    检查条目是否完整。
    """
    if entry is None: return True  # 已被标记删除的视为"处理完毕"

    # 1. 检查 Prompt 和 RCoT 是否为空
    if not entry.get("sci-RCoT") or not entry.get("science_abstract_prompt"):
        return False

    # 2. 检查 Reasoning
    reasoning = entry.get("reasoning")
    if not reasoning:
        return False

    # 3. 检查 Reasoning 内部内容
    valid_keys = ["ScientificLaw", "EntityStructure", "ScientificProcess"]
    has_content = False

    for key in valid_keys:
        val = reasoning.get(key)
        if val:
            has_content = True
            break

    return has_content


# ============ 复用核心函数 (保持一致性) ============

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def clean_json_output(text):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r"^```(json)?\s*", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx: end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                json_str_fixed = re.sub(r",\s*\}", "}", json_str)
                return json.loads(json_str_fixed)
            except:
                return None
    return None


# --- Prompt 生成函数 ---

def generate_reasoning(data):
    system_prompt = """你是一名科学图像推理标注员。你的职责是：基于输入的论文文本与图片，只提取图像与图注明确支持的科学信息，输出严格结构化的 reasoning JSON。禁止编造超出文本与图像所支持的结论；不得引入图像中不可见或图注未支持的元素；不得进行常识性补全。内部可以先思考并对图像整体布局做隐式理解，但最终输出必须仅包含指定的 JSON 字段。
详细要求：
1) 内部先解析输入，解析图片整体布局与主要视觉元素，优先依据信息源的权重：图片>figure_title > 图注 > 文章正文。注意：该步骤仅用于内部推理，最终输出不单列此描述。
2) 仅根据“推理能力”中被选中的标签，完成对应信息的“提取”与“可视化描述”：
- ScientificLaw（科学规律）
提取：定律、原理、边界条件、适用前提等科学约束（只提取能从图与图注直接支持的）。
可视化：每项约束在图中的体现（几何布局、标注、符号、测量元素等）。
- EntityStructure（科学实体）
提取：图片中的科学实体名称（名词化，不含动词）。
可视化：每个实体的形态结构、颜色/材质/纹理、空间位置与尺度、数量等视觉特性。
- ScientificProcess（科学过程）
提取：图片中的过程名称（如制备流程、实验步骤、时间序列、机制链条）。
可视化：这个过程不同阶段的状态、变化条件（如箭头、时间轴、循环）的视觉体现。
3) 输出要求：
- 请用英文回答。
- 严格遵循下方 JSON 结构，不能输出任何额外自由文本。
- terms：仅写术语/名词（不含句子、标点与修饰），按图注与图像确证的粒度。
- visualization：对于terms中的每一项都要有一段可视化描述；使用具体、可还原画面的视觉要素描述；不得引入图外信息。
- 未被选择的能力键必须为 null。
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
    - text：{{"article_title": "{data['article_title']}", "article_abstract": "{data['article_abstract']}", "article_body": "{data['article_body']}", "figure_title": "{data['figure_title']}"}}
    - figure_caption：{{"figure_caption": "{data['figure_caption']}"}}
    - reasoning_ability：{data["segments"]["labels"]}
    - subject：{data["subjects"]}
    """
    result = call_model(client_A, API_A_MODEL, system_prompt, user_prompt, temperature=0.1, image_base64=image_base64)
    reasoning_json = clean_json_output(result)
    if reasoning_json is None:
        reasoning_json = {"reasoning": {}}
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
    raw_result = call_model(client_A, API_A_MODEL, system_prompt, user_prompt, temperature=0.3,
                            image_base64=image_base64)
    result_json = clean_json_output(raw_result)

    if result_json is None:
        print(f"\n[DEBUG CoT Fail] {data.get('figure_title', 'No Title')}\nRaw Output snippet: {raw_result[:200]}...")
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
    if prompt_json is None: return "", []
    final_prompt = prompt_json.get("abstract_prompt", "")
    retained_text_list = prompt_json.get("retained_text", [])
    return final_prompt, retained_text_list


# ============ 修复任务逻辑 ============
def is_reasoning_valid(reasoning_data):
    """辅助函数：检查 reasoning 是否有效"""
    if not reasoning_data:
        return False
    valid_keys = ["ScientificLaw", "EntityStructure", "ScientificProcess"]
    for key in valid_keys:
        item = reasoning_data.get(key)
        if item and (item.get("terms") or item.get("visualization")):
            return True
    return False


def process_repair_task(index, task_data, manager):
    """
    执行单条数据的修复流程，包含重试和删除逻辑
    """
    seg = task_data["segments"]
    key = seg.get("path")

    if not os.path.exists(key):
        return f"Skip: {key} (File not found)"

    try:
        # === 逻辑重试循环 ===
        for attempt in range(MAX_LOGIC_RETRIES):
            # Step 1: 获取 Reasoning
            # 如果是第 0 次尝试，优先复用；如果是第 1+ 次（重试中），强制重新生成
            current_entry = manager.data_list[index]
            # 注意：current_entry 可能是 None 如果其他线程操作过（虽然这里是一对一），加个判断
            if current_entry is None:
                return f"Stopped: {key} (Entry deleted by other process)"

            existing_reasoning = current_entry.get("reasoning")
            reasoning = {}

            if attempt == 0 and is_reasoning_valid(existing_reasoning):
                reasoning = {"reasoning": existing_reasoning}
            else:
                # 重新生成
                reasoning = generate_reasoning(task_data)

            # 校验 Reasoning
            if not is_reasoning_valid(reasoning.get("reasoning")):
                if attempt < MAX_LOGIC_RETRIES - 1:
                    continue  # 重试
                else:
                    break  # 失败，准备删除

            # Step 2: 生成 CoT
            sci_rcot, rendered_text_list = generate_cot(reasoning, task_data)
            if not sci_rcot:
                if attempt < MAX_LOGIC_RETRIES - 1:
                    continue
                else:
                    break

            # Step 3: 生成 Prompt
            abstract_prompt, retained_text_list = generate_prompt(reasoning, sci_rcot, rendered_text_list)
            if not abstract_prompt:
                if attempt < MAX_LOGIC_RETRIES - 1:
                    continue
                else:
                    break

            # === 成功路径 ===
            result_entry = {
                "image_path": key,
                "reasoning": reasoning.get("reasoning", {}),
                "sci-RCoT": sci_rcot,
                "rendered_text_stage2": rendered_text_list,
                "science_abstract_prompt": abstract_prompt,
                "retained_text_stage3": retained_text_list
            }
            manager.update_entry(index, result_entry)
            return f"Repaired: {key}"

        # === 失败路径 (循环结束仍未 return) ===
        # 执行删除操作
        manager.delete_entry_and_file(index, key)
        return f"DELETED: {key} (Failed {MAX_LOGIC_RETRIES} attempts)"

    except Exception as e:
        import traceback
        return f"Error: {key} ({str(e)})"


# ============ 主程序 ============

manager = RepairManager(OUTPUT_PATH, batch_size=BATCH_SIZE)


def signal_handler(sig, frame):
    print("\n[System] 捕获中断信号，保存中...")
    manager.force_save()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
atexit.register(manager.force_save)

if __name__ == "__main__":
    print(f"开始修复流程...")
    print(f"目标文件: {OUTPUT_PATH}")

    # 1. 扫描待修复的索引和图片路径
    indices_to_fix = []

    # 遍历时跳过已经是 None 的数据
    for idx, entry in enumerate(manager.data_list):
        if entry is not None and not is_entry_complete(entry):
            indices_to_fix.append(idx)

    if not indices_to_fix:
        print("所有数据完整性检查通过，无需修复。")
        sys.exit(0)

    print(f"发现 {len(indices_to_fix)} 条不完整数据。")
    print(f"正在加载原始输入数据以获取上下文: {CLASSIFIED_INPUT}")

    # 2. 加载原始数据
    raw_items = json.load(open(CLASSIFIED_INPUT, "r", encoding="utf-8"))
    path_to_task_map = {}

    for item in raw_items:
        segs = item.get("segments", [])
        if isinstance(segs, dict): segs = [segs]

        for seg in segs:
            filename = seg.get("filename")
            if not filename:
                raw_path = seg.get("path")
                if raw_path: filename = os.path.basename(raw_path)

            if filename:
                full_path = os.path.join(IMAGE_DIR, filename)
                seg["path"] = full_path
                task = item.copy()
                task["segments"] = seg
                path_to_task_map[full_path] = task

    # 3. 准备修复任务
    repair_tasks = []

    for idx in indices_to_fix:
        entry = manager.data_list[idx]
        if entry is None: continue

        img_path = entry["image_path"]
        task_data = path_to_task_map.get(img_path)

        if task_data:
            repair_tasks.append((idx, task_data))
        else:
            print(f"[Warn] 无法找到原始输入数据，标记删除: {img_path}")
            # 如果连原始数据都找不到，也建议直接删了，防止卡流程
            manager.delete_entry_and_file(idx, img_path)

    print(f"成功构建 {len(repair_tasks)} 个修复任务。开始执行...")

    # 4. 多线程执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_repair_task, idx, task, manager): idx
            for idx, task in repair_tasks
        }

        for future in tqdm(as_completed(future_to_idx), total=len(repair_tasks), desc="Repairing"):
            idx = future_to_idx[future]
            try:
                msg = future.result()
                if "Failed" in msg or "Error" in msg or "DELETED" in msg:
                    print(f"\n{msg}")
            except Exception as exc:
                print(f"\n[Fatal Error] Idx {idx}: {exc}")

    manager.force_save()
    print("修复完成。")