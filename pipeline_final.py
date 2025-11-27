"""
pipeline_dual_api_for_scientific_image_reasoning.py

功能：
- 阶段1+2：使用 API_A（多模态） -> 生成 reasoning + sci-RCoT
- 阶段3：使用 API_B（文本） -> 生成 science_abstract_prompt + mapping_log
"""

import re
import json
import base64
import os
from openai import OpenAI
from pathlib import Path

# ============ 配置区域 ============
# --- 第一组 API：用于阶段1+2 ---
# API_A_KEY = "sk-WJMFOYf7LoWps17qnYlgNMIVetyEcFTQZhacY5aAKkjHFZM6"
# API_A_BASE = "https://chat.intern-ai.org.cn/api/v1/"
# API_A_MODEL = "internvl3.5-241b-a28b"
API_A_KEY = "替换自己的key"
API_A_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_A_MODEL = "qwen3-vl-plus"

# --- 第二组 API：用于阶段3 ---
# API_B_KEY = "替换自己的key"
# API_B_BASE = "https://chat.intern-ai.org.cn/api/v1/"
# API_B_MODEL = "internvl3.5-latest"
API_B_KEY = "替换自己的key"
API_B_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_B_MODEL = "qwen3-vl-plus"

# --- 文件路径 ---
# 原 INPUT_PATH 保留为备用
#INPUT_PATH = "D:\\code\\project\\classified_metadata_1a.json"
# 批量输入文件（classified metadata）
CLASSIFIED_INPUT = "D:\\code\\project\\classified_metadata_1a.json"
OUTPUT_PATH = "D:\\code\\project\\output_dataset_qwen3-vl-plus_qwen3-vl-plus.json"
# 缓存文件，用于断点续跑
CACHE_PATH = "D:\\code\\project\\pipeline_cache.json"

# =================================

client_A = OpenAI(api_key=API_A_KEY, base_url=API_A_BASE)
client_B = OpenAI(api_key=API_B_KEY, base_url=API_B_BASE)


# ============ 工具函数 ============

def encode_image(image_path):
    """将图片转为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_model(client, model, system_prompt, user_prompt, temperature=0.1, top_p=1.0, image_base64=None):
    """
    统一模型调用：
    - 若传入 image_base64，则启用多模态消息格式；
    - 若未传入图片，则为纯文本。
    """
    if image_base64:
        # 多模态输入格式
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                #qwen
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
                ]
            }
        ]
    else:
        # 纯文本输入格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        messages=messages
    )
    return response.choices[0].message.content



def clean_json_output(text):
    """去除 markdown 包裹并返回 JSON"""
    text = text.strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except:
        return None


# ============ 阶段一：生成 reasoning ============
def generate_reasoning(data):
    system_prompt = """你是一名科学图像推理标注员。你的职责是：基于输入的论文文本与图片，只提取图像与图注明确支持的科学信息，输出严格结构化的 reasoning JSON。禁止编造超出文本与图像所支持的结论；不得引入图像中不可见或图注未支持的元素；不得进行常识性补全。内部可以先思考并对图像整体布局做隐式理解，但最终输出必须仅包含指定的 JSON 字段。
详细要求：
1) 内部先解析输入，解析图片整体布局与主要视觉元素，优先依据信息源的权重：图片>figure_title > 图注 > 文章正文。注意：该步骤仅用于内部推理，最终输出不单列此描述。
2) 仅根据“推理能力”中被选中的标签，完成对应信息的“提取”与“可视化描述”：
   - ScientificConsistency（科学一致性）
     提取：定律、原理、边界条件、适用前提等科学约束（只取能从图与图注直接支持的）。
     可视化：每项约束在图中的体现（几何布局、标注、符号、测量元素等）。
   - EntityStructure（科学实体）
     提取：图片中的科学实体名称（名词化，不含动词）。
     可视化：每个实体的形态结构、颜色/材质/纹理、空间位置与尺度、数量等视觉特性。
   - VisualizationStyle（渲染风格）
     提取：图片渲染/制图风格术语（如示意图/显微图/热力图/轨迹叠加/三维渲染等）。
     可视化：该风格在图中的具体呈现方式，以及如何服务科学表达（例：颜色映射、箭头/等值线/误差棒等）。
   - ScientificProcess（科学过程）
     提取：过程/阶段名称（如制备流程、实验步骤、时间序列阶段）。
     可视化：阶段序列、状态变化、因果或符号化元素（箭头、时间轴、循环）的视觉体现。
3) 输出要求：
- 请用英文回答。
- 严格遵循下方 JSON 结构，不能输出任何额外自由文本。
- terms：仅写术语/名词（不含句子、标点与修饰），按图注与图像确证的粒度。
- visualization：与 terms 一一对应；使用具体、可还原画面的视觉要素描述；不得引入图外信息。
- 已选中的能力若图中确无可提取内容，可将 terms 与 visualization 设为 []（空数组）。
- 未被选择的能力键必须为 null。"""

    image_base64 = encode_image(data["segments"]["path"])
    user_prompt = f"""
输入：
- text：{{"article_title": "{data['article_title']}", "article_abstract": "{data['article_abstract']}", "article_body": "{data['article_body']}", "figure_title": "{data['figure_title']}"}}
- figure_caption：{{"figure_caption": "{data['figure_caption']}"}}
- reasoning_ability：{data["segments"]["labels"]}
- subject：{data["subjects"]}

输出格式：
{{
  "reasoning": {{
    "ScientificConsistency": {{"terms": [], "visualization": []}},
    "EntityStructure": {{"terms": [], "visualization": []}},
    "VisualizationStyle": {{"terms": [], "visualization": []}},
    "ScientificProcess": {{"terms": [], "visualization": []}}
  }}
}}
"""
    result = call_model(client_A, API_A_MODEL, system_prompt, user_prompt, temperature=0.1,image_base64=image_base64)
    reasoning_json = clean_json_output(result)
    if reasoning_json is None:
        print("Reasoning 解析失败，原始输出：", result)
        reasoning_json = {"reasoning": {}}
    return reasoning_json


# ============ 阶段二：生成 sci-RCoT ============
def generate_cot(reasoning_json, data):
    system_prompt = """你是一名科学图像可视化复述员。你的职责是：输入reasoning，以reasoning中的 visualization 条目为主，参考输入的图片，输出连贯、细节充分的场景化描述sci-RCoT。
要求：sci-RCoT要完整覆盖reasoning 中各已选标签的 visualization 数组里每一条要点，不得遗漏、不得改写其含义，严禁新增任何reasoning、图注或图像中未出现的要素；语言风格要具象、连贯、可据文还原画面，避免学术化抽象术语与不确定词。"""
    image_base64 = encode_image(data["segments"]["path"])
    user_prompt = f"""
输入：
- figure_caption：{{"figure_title": "{data['figure_title']}", "figure_caption": "{data['figure_caption']}"}}
- reasoning：{json.dumps(reasoning_json, ensure_ascii=False)}

输出：请用英文回答，仅输出一段 sci-RCoT。
"""
    return call_model(client_A, API_A_MODEL, system_prompt, user_prompt, temperature=0.3,image_base64=image_base64)


# ============ 阶段三：生成 science_abstract_prompt ============
def generate_prompt(reasoning_json, sci_rcot):
    system_prompt = """你是科学图像推理生成prompt助手。你的职责是：输入sci-RCoT和各个推理能力维度的terms和visualization。在输入sci-RCoT基础上，你需要用每个推理能力维度terms替换sci-RCoT直白的描述。输出JSON类型的science_abstract_prompt。
要求：请用英文回答，science_abstract_prompt 是一条简洁的prompt，也是一个语义压缩版的sci-RCoT，要求内容只来源于terms，并且限定科学闭合，替换必须符合原物理语义，不得引入新机制。"""
    user_prompt = json.dumps({
        "reasoning": reasoning_json["reasoning"],
        "sci-RCoT": sci_rcot
    }, ensure_ascii=False, indent=2)

    result = call_model(client_B, API_B_MODEL, system_prompt, user_prompt, temperature=0.3,image_base64=None)
    prompt_json = clean_json_output(result)
    if prompt_json is None:
        print("Prompt 阶段输出非JSON，原始输出：", result)
        prompt_json = {"science_abstract_prompt": ""}
    return prompt_json


# ============ 主流程（已改为批量处理，带缓存） ============
if __name__ == "__main__":
    # 读取批量输入
    classified_path = Path(CLASSIFIED_INPUT)
    if not classified_path.exists():
        print(f"批量输入文件 {CLASSIFIED_INPUT} 不存在，退出。")
        raise SystemExit(1)

    raw_items = json.load(open(classified_path, "r", encoding="utf-8"))
    if not isinstance(raw_items, list):
        print(f"{CLASSIFIED_INPUT} 不是一个 JSON 数组，请检查输入格式。")
        raise SystemExit(1)

    # 将每个 top-level item 的 segments 展开为独立任务（兼容 segments 为 dict / list / 空）
    tasks = []
    for item in raw_items:
        segs = item.get("segments", [])
        if isinstance(segs, dict):
            segs = [segs]
        if not segs:
            # 若没有 segments，尝试使用 top-level local_path（如果有）
            lp = item.get("local_path") or item.get("image_path") or item.get("image_url")
            if lp:
                segs = [{"path": lp, "filename": Path(lp).name, "width": item.get("image_width"), "height": item.get("image_height"), "labels": []}]
            else:
                # 无可用图像信息，跳过该条目
                continue
        for seg in segs:
            # 构造 task：保留 top-level 元信息并把单个 segment 放入 segments 字段
            task = {
                "image_id": item.get("image_id"),
                "local_path": item.get("local_path"),
                "article_title": item.get("article_title", ""),
                "article_abstract": item.get("article_abstract", ""),
                "article_body": item.get("article_body", ""),
                "figure_title": item.get("figure_title", ""),
                "figure_caption": item.get("figure_caption", ""),
                "subjects": item.get("subjects", []),
                "segments": seg
            }
            tasks.append(task)

    # 加载缓存（若存在），以便断点续跑
    cache_file = Path(CACHE_PATH)
    cache = {"processed": [], "outputs": []}
    if cache_file.exists():
        try:
            cache = json.load(open(cache_file, "r", encoding="utf-8"))
        except Exception as e:
            print("加载缓存失败，忽略并重建缓存：", e)
            cache = {"processed": [], "outputs": []}

    processed = cache.get("processed", [])
    outputs = cache.get("outputs", [])
    processed_set = set(processed)

    total = len(tasks)
    for idx, data in enumerate(tasks):
        # 使用 segment path 或 filename 或 image_id 作为唯一键
        seg = data.get("segments", {})
        key = seg.get("path") or seg.get("filename") or data.get("image_id") or f"task_{idx}"

        if key in processed_set:
            print(f"[{idx+1}/{total}] 跳过已处理项: {key}")
            continue

        print(f"[{idx+1}/{total}] 处理项: {key}")
        try:
            img_path = seg.get("path")
            # 如果图片不存在，记录并跳过（同时加入 processed，避免重复尝试）
            if not img_path or not Path(img_path).exists():
                print(f"警告：图像文件不存在，跳过：{img_path}")
                outputs.append({
                    "image_path": img_path,
                    "reasoning": {},
                    "sci-RCoT": "",
                    "science_abstract_prompt": "",
                    "error": "image_not_found"
                })
                processed.append(key)
                processed_set.add(key)
                with open(CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump({"processed": processed, "outputs": outputs}, f, ensure_ascii=False, indent=2)
                continue

            # 阶段1
            reasoning = generate_reasoning(data)
            # 阶段2
            sci_rcot = generate_cot(reasoning, data)
            # 阶段3
            prompt_result = generate_prompt(reasoning, sci_rcot)

            final_output = {
                "image_path": img_path,
                "reasoning": reasoning.get("reasoning", {}),
                "sci-RCoT": sci_rcot,
                "science_abstract_prompt": prompt_result.get("science_abstract_prompt")
            }

            outputs.append(final_output)
            processed.append(key)
            processed_set.add(key)

            # 每处理完一项就保存缓存，避免中断丢失进度
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"processed": processed, "outputs": outputs}, f, ensure_ascii=False, indent=2)

            print(f"[{idx+1}/{total}] 完成：{key}，已保存缓存。")

        except Exception as e:
            # 出错时保存当前缓存后抛出以便人工检查
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"processed": processed, "outputs": outputs}, f, ensure_ascii=False, indent=2)
            print(f"[{idx+1}/{total}] 处理失败，已保存缓存。错误：{e}")
            raise

    # 全部完成后，将聚合结果写入 OUTPUT_PATH（覆盖）
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    # 可选：删除缓存文件（仅在完全成功时）
    try:
        cache_file.unlink()
    except Exception:
        pass

    print(f"全流程完成，聚合结果已保存到 {OUTPUT_PATH}")
