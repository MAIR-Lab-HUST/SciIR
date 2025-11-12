"""
用途：
- 实现三阶段自动化调用大模型：
  stage1: 图片+文本 -> reasoning JSON（严格结构）
  stage2: 基于 reasoning.visualization -> sci-RCoT（单段描述）
  stage3: sci-RCoT + reasoning -> 输出 prompt 与 specific_mapping_log
- 最终将每个样本保存为一条 JSONL（包含 image_path, reasoning, sci_RCoT, prompt, mapping_log）
- 使用本地文件上传方式传递图片（稳定）

"""
import re
import json
import os
import dashscope
from dashscope import MultiModalConversation, Generation
from pathlib import Path

# ============ 配置区域 ============
# 请确保已设置环境变量 DASHSCOPE_API_KEY
# export DASHSCOPE_API_KEY="sk-xxxxxxxx"
# =================================

# 输入数据文件路径
input_path = "./scir_dataset/classified_metadata.json"
# 输出结果路径
output_path = "./scir_dataset/caption.jsonl" # 修改为 .jsonl 以反映输出格式


# ============ 工具函数 (已修改) ============

def get_file_url(abs_path):
    """
    将一个绝对文件路径转换为 file:// URL。
    """
    if os.name == 'nt':  # Windows
        # Windows 路径的正确格式是 file://D:/path/to/file
        file_url = f"file://{abs_path.replace(os.sep, '/')}"
    else:  # Linux/macOS
        file_url = f"file://{abs_path}"
    return file_url

def call_model(system_prompt, user_prompt, model, image_abs_path=None, temperature=0.1, top_p=1.0):
    """
    统一封装的 Dashscope 模型调用函数 (qwen3-vl-plus 和 qwen3-max)
    接收一个可选的绝对图像路径。
    """
    api_key = ""
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

    if model == "qwen3-vl-plus":
        # qwen-vl 模型没有独立的 system role，我们将 system_prompt 合并到 user_prompt 中
        combined_prompt_text = f"{system_prompt}\n\n{user_prompt}"

        user_content = [{"text": combined_prompt_text}]
        if image_abs_path:
            file_url = get_file_url(image_abs_path)
            user_content.insert(0, {"image": file_url})

        messages = [{"role": "user", "content": user_content}]

        response = MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content[0]['text']
        else:
            raise RuntimeError(f"API call failed for {model}: {response.code} - {response.message}")

    elif model == "qwen3-max":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = Generation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format="message",
            temperature=temperature,
            top_p=top_p
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise RuntimeError(f"API call failed for {model}: {response.code} - {response.message}")
    else:
        raise ValueError(f"不支持的模型: {model}")


# ============ 阶段一：生成 reasoning  ============
def generate_reasoning(data, image_abs_path):
    system_prompt = """你是一名科学图像推理标注员。你的职责是：基于给定论文文本与图片，只提取图像与图注明确支持的科学信息，生成严格结构化的 reasoning JSON。禁止编造超出文本与图像所支持的结论；不得引入图像中不可见或图注未支持的元素；不得进行常识性补全。
详细要求：
1) 内部先解析输入，依据信息源的权重：图片>图注>文章正文。注意：该步骤仅用于内部推理，最终输出不单列此描述。
2) 仅根据“推理能力”中被选中的标签，完成对应信息的“提取”与“可视化描述”：
   - ScientificConsistency（科学一致性）
     提取：定律、原理、边界条件、适用前提等科学约束（只取能从图与图注直接支持的）。
     可视化：每项约束在图中的体现（几何布局、标注、符号、测量元素等）。
   - EntityStructure（科学实体）
     提取：图片中的科学实体名称（名词化，不含动词）。
     可视化：每个实体的形态结构、颜色/材质/纹理、空间位置与尺度、数量等视觉特性。
   - ScientificProcess（科学过程）
     提取：过程/阶段名称（如制备流程、实验步骤、时间序列阶段）。
     可视化：阶段序列、状态变化、因果或符号化元素（箭头、时间轴、循环）的视觉体现。
3) 输出要求：
- 严格遵循下方 JSON 结构，不能输出任何额外自由文本。
- terms：仅写术语/名词（不含句子、标点与修饰），按图注与图像确证的粒度。
- visualization：与 terms 一一对应；使用具体、可还原画面的视觉要素描述；不得引入图外信息。
- 已选中的能力若图中确无可提取内容，可将 terms 与 visualization 设为 []（空数组）。
- 未被选择的能力键必须为 null。"""

    user_prompt = f"""
输入：
- 文本：{{"article_title": "{data.get('article_title', '')}", "article_abstract": "{data.get('article_abstract', '')}", "article_body": "{data.get('article_body', '')}"}}
- 图注：{{"figure_title": "{data.get('figure_title', '')}", "figure_caption": "{data.get('figure_caption', '')}"}}
- 图片：已通过文件路径上传
- 推理能力：{data.get('capabilities_list', [])}
- 主题：{data.get('subjects', [])}

请严格输出以下 JSON 结构：
{{
  "reasoning": {{
    "ScientificConsistency": {{"terms": [], "visualization": []}},
    "EntityStructure": {{"terms": [], "visualization": []}},
    "ScientificProcess": {{"terms": [], "visualization": []}}
  }}
}}
"""

    result = call_model(
        system_prompt,
        user_prompt,
        model="qwen3-vl-plus",
        image_abs_path=image_abs_path,
        temperature=0.1,
        top_p=1.0
    )

    result_clean = result.strip()
    result_clean = re.sub(r"^```json\s*|\s*```$", "", result_clean, flags=re.DOTALL).strip()

    try:
        return json.loads(result_clean)
    except json.JSONDecodeError as e:
        print(f"Reasoning 解析失败，错误: {e}")
        print("原始输出：", result)
        return {"reasoning": {}}


# ============ 阶段二：生成 sci-RCoT  ============
def generate_cot(reasoning_json, data, image_abs_path):
    system_prompt = """你是一名科学图像可视化复述员。你的职责是：输入reasoning，以reasoning中的visualization条目为主，参考图片，输出连贯、细节充分的场景化描述sci-RCoT。要求：sci-RCoT要完整覆盖reasoning中各已选标签的visualization数组里每一条要点，不得遗漏、不得改写其含义，严禁新增任何reasoning、图注或图像中未出现的要素；语言风格要具象、连贯、可据文还原画面，避免学术化抽象术语与不确定词。
"""

    user_prompt = f"""
输入：
- 图片：已通过文件路径上传
- reasoning：{json.dumps(reasoning_json, ensure_ascii=False)}

输出：仅输出一段sci-RCoT。
"""

    return call_model(
        system_prompt,
        user_prompt,
        model="qwen3-vl-plus",
        image_abs_path=image_abs_path,
        temperature=0.3,
        top_p=1.0
    )


# ============ 阶段三：生成 prompt ============
def generate_prompt(reasoning_json, sci_rcot):
    """
    生成 prompt 和 specific_mapping_log
    这个阶段不需要图片
    """
    system_prompt = """你是科学图像推理生成prompt助手。你的职责是：输入sci-RCoT和各个推理能力维度的terms和visualization。在输入sci-RCoT基础上，结合每个推理能力维度terms和visualization，用抽象的科学语言直接替换sci-RCoT直白的描述。输出JSON类型的prompt和specific_mapping_log。
要求：prompt 是一个语义压缩版的CoT。要求限定科学闭合，替换必须符合原物理语义，不得引入新机制；specific_mapping_log 逐项记录每个推理能力的原描述替换抽象科学描述对照关系。不要求每个推理能力都要相应的替换，如果原始描述与抽象后的描述文字上大致相似请去掉并设置为none。"""
    user_prompt = json.dumps({
        "reasoning": reasoning_json["reasoning"],
        "sci-RCoT": sci_rcot
    }, ensure_ascii=False, indent=2)

    result = call_model(
        system_prompt,
        user_prompt,
        model="qwen3-max",
        temperature=0.3,
        top_p=1.0
    )

    result_clean = result.strip()
    result_clean = re.sub(r"^```json\s*|\s*```$", "", result_clean, flags=re.DOTALL).strip()

    try:
        return json.loads(result_clean)
    except json.JSONDecodeError as e:
        print(f"prompt 阶段解析失败，错误: {e}")
        print("原始输出：", result)
        return {"prompt": "", "specific_mapping_log": {}}


# ============ 主流程  ============
def process_single_sample(data):
    """
    处理单个样本的完整流程
    """
    try:
        parent_folder = 'scir_dataset'
        # 1. 仅在此处构建和验证路径
        full_path = os.path.join(parent_folder, data["image_path"])
        abs_path = os.path.abspath(full_path)

        # 验证图片路径是否存在
        if not os.path.exists(abs_path):
            print(f"警告：图片文件不存在，已跳过 - {abs_path}")
            return None

        print(f"处理图片: {data['image_path']}")

        # 2. 将验证后的绝对路径传递下去
        print("阶段一：生成 reasoning ...")
        reasoning = generate_reasoning(data, abs_path)

        print("阶段二：生成 sci-RCoT ...")
        sci_rcot = generate_cot(reasoning, data, abs_path)

        print("阶段三：生成 prompt ...")
        prompt_result = generate_prompt(reasoning, sci_rcot)

        # 构建最终输出
        final_output = {
            "image_path": data["image_path"],
            "reasoning": reasoning.get("reasoning", {}),
            "sci-RCoT": sci_rcot,
            "prompt": prompt_result.get("prompt", ""),
            "specific_mapping_log": prompt_result.get("specific_mapping_log", {})
        }

        return final_output

    except Exception as e:
        print(f"处理样本 {data.get('image_path', 'N/A')} 时出错: {e}")
        return None


def main():
    # 读取输入数据
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 如果输入是单个样本的字典，将其放入列表中以便统一处理
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    # ==================== 数据转换 ====================
    # 将原始数据转换为以每个 segment 为单位的独立样本列表
    all_samples = []
    print("正在转换输入数据...")
    for item in raw_data:
        # 确定图注内容
        caption_text = item.get("figure_caption", "")
        # ================================================

        # 为每个 segment 创建一个样本
        for segment in item.get("segments", []):
            if "path" not in segment or "labels" not in segment:
                continue # 跳过不完整的 segment

            sample = {
                # 从顶层继承的共享信息
                "article_title": item.get("article_title", ""),
                "article_abstract": item.get("article_abstract", ""),
                "article_body": item.get("article_body", ""),
                "figure_title": item.get("figure_title", ""),
                "subjects": item.get("subjects", []),
                # 处理后的图注
                "figure_caption": caption_text,
                # 来自 segment 的特定信息
                "image_path": segment["path"],
                "capabilities_list": segment["labels"]
            }
            all_samples.append(sample)

    print(f"数据转换完成，共生成 {len(all_samples)} 个待处理样本。")
    # ================================================

    # 处理所有转换后的样本
    results = []
    for idx, sample in enumerate(all_samples, 1):
        print(f"\n===== 处理第 {idx}/{len(all_samples)} 个样本 =====")
        result = process_single_sample(sample)
        if result:
            results.append(result)

    # 保存为 JSONL 格式
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\n全部处理完成，共 {len(results)} 个样本，结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
