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

# 输入输出路径
CLASSIFIED_INPUT = "scir_dataset/classified_abstracts3.json"
OUTPUT_PATH = "scir_dataset/caption3.json"
CACHE_PATH = "scir_dataset/caption_cache3.json"

# [修改 1] 新增图片文件夹路径配置
IMAGE_DIR = "scir_dataset/filtered_images_3"

# 并发配置
MAX_WORKERS = 20
MAX_RETRIES = 6
RETRY_DELAY = 2

# 批量写入配置
BATCH_SIZE = 100  # 每积攒多少条数据写入一次磁盘

client_A = OpenAI(api_key=API_A_KEY, base_url=API_A_BASE)
client_B = OpenAI(api_key=API_B_KEY, base_url=API_B_BASE)


# ============ 数据管理类 ============

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
                    # 兼容性检查：确保结构正确
                    if "processed" not in self.data: self.data["processed"] = []
                    if "outputs" not in self.data: self.data["outputs"] = []
            except json.JSONDecodeError:
                print(f"[Warn] 缓存文件 {save_path} 损坏，将重置。")
                self.data = {"processed": [], "outputs": []}
        else:
            self.data = {"processed": [], "outputs": []}

        # 为了快速查找，维护一个 set
        self.processed_set = set(self.data["processed"])

    def is_processed(self, key):
        return key in self.processed_set

    def add_result(self, key, result_entry):
        """线程安全地添加结果并自动触发批量保存"""
        with self.lock:
            self.data["processed"].append(key)
            self.data["outputs"].append(result_entry)
            self.processed_set.add(key)
            self.unsaved_count += 1

            # 达到批量阈值，触发保存
            if self.unsaved_count >= self.batch_size:
                self._save_to_disk_unsafe()
                self.unsaved_count = 0

    def force_save(self):
        """强制保存当前所有数据（用于程序退出或异常时）"""
        with self.lock:
            if self.unsaved_count > 0:
                print(f"\n[System] 正在强制保存剩余的 {self.unsaved_count} 条数据...")
                self._save_to_disk_unsafe()
                self.unsaved_count = 0
            else:
                # 即使没有新数据，如果文件不存在也要创建
                if not os.path.exists(self.save_path):
                    self._save_to_disk_unsafe()

    def _save_to_disk_unsafe(self):
        """
        内部保存方法，使用原子写入防止文件损坏。
        调用前必须持有锁。
        """
        temp_path = self.save_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            # 原子操作：替换原文件
            os.replace(temp_path, self.save_path)
        except Exception as e:
            print(f"[Error] 保存文件失败: {e}")


# ============ 工具函数 ============

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
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ============ 核心 Prompt 与逻辑 ============

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


# ============ 单个任务处理 ============

def process_single_task(task_data, manager: ResultManager):
    """
    处理单个任务，并调用 manager 进行结果管理
    """
    seg = task_data["segments"]
    key = seg.get("path") or seg.get("filename")

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


# ============ 主程序 ============

# 初始化全局管理器
manager = ResultManager(CACHE_PATH, batch_size=BATCH_SIZE)


# 注册信号处理，确保中断时保存
def signal_handler(sig, frame):
    print("\n[System] 捕获中断信号，正在保存数据，请勿强制关闭...")
    manager.force_save()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
atexit.register(manager.force_save)

if __name__ == "__main__":
    print(f"初始化... 图片目录: {IMAGE_DIR}")
    print(f"缓存路径: {CACHE_PATH}, 批量写入大小: {BATCH_SIZE}")

    classified_path = Path(CLASSIFIED_INPUT)
    if not classified_path.exists():
        print(f"输入文件未找到: {CLASSIFIED_INPUT}")
        exit(1)

    raw_items = json.load(open(classified_path, "r", encoding="utf-8"))

    # --- 任务构建 ---
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
            # [修改 2] 路径重组逻辑
            # 1. 获取文件名 (优先使用 filename 字段，如果没有则从 path 中提取)
            filename = seg.get("filename")
            if not filename:
                raw_path = seg.get("path")
                if raw_path:
                    filename = os.path.basename(raw_path)

            # 2. 如果成功获取文件名，则拼接新的 IMAGE_DIR
            if filename:
                new_full_path = os.path.join(IMAGE_DIR, filename)
                seg["path"] = new_full_path
                seg["filename"] = filename  # 确保 filename 字段也正确
            else:
                # 如果完全无法获取文件名，跳过或保留原样(视具体需求，这里保留原样但会打印警告)
                # print(f"[Warn] 无法解析文件名: {seg}")
                pass

            task = item.copy()
            task["segments"] = seg
            tasks.append(task)

    # --- 过滤已完成任务 ---
    pending_tasks = []
    for idx, data in enumerate(tasks):
        seg = data["segments"]
        # key 优先使用 path (现在是包含 filtered_images3 的完整路径)
        key = seg.get("path") or seg.get("filename") or str(idx)
        if not manager.is_processed(key):
            pending_tasks.append(data)

    total = len(tasks)
    processed = len(manager.data["processed"])
    print(f"总任务: {total}, 已完成: {processed}, 待处理: {len(pending_tasks)}")

    # --- 并发执行 ---
    if pending_tasks:
        print(f"开始处理，线程数: {MAX_WORKERS}...")

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

    print(f"所有任务执行完毕。最终结果将保存至: {OUTPUT_PATH}")
    manager.force_save()

    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(manager.data["outputs"], f, ensure_ascii=False, indent=2)
        print("最终结果已导出。")
    except Exception as e:
        print(f"导出最终结果失败: {e}")