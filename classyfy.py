import os
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from collections import defaultdict
import time

# ✅ 初始化客户端
client = OpenAI(
    api_key="sk-eGILuyIosedfyzKZB2IMycqSAaywcJpgbzpi3EGxuzmj53mM",
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

# 分类系统提示词
classification_prompt = """
角色与任务
你是一名严谨的科学图像分类器。你的任务是围绕下述四个维度，对输入图像进行多标签、多维度的相关度评分（1–10）。
评价维度与要点
ScientificConsistency
判定图像是否涉及并呈现与学科规律与约束相关的要素（如价态/键连、尺度关系、能量/动量守恒的可视暗示）。注意：评分为该维度的相关度，而非正确度。
EntityStructure
判定图像是否涉及科学实体的结构与几何关系（如分子/晶格/细胞/星系/仪器组件的形态、连接、拓扑与相对尺度）。评分为相关度。
ScientificProcess
判定图像是否呈现过程性信息（时间演化、因果链、状态转变、反应机理）或存在明确过程线索（不同阶段标签、时间轴）。评分为相关度。

评分原则
分数含义：1–10 表示该维度在图像中的"相关度"，非表达强度、非正确度、非质量分。
评分锚点（相关度）：
1–2：几乎不涉及
3–4：有零星线索，但很弱
5–6：中等相关，有一定证据
7–8：强相关，证据充分
9–10：主导性相关，是图像核心

证据与限制
仅基于图像中可见且清晰的证据作答；不臆测不可见要素。
维度名仅限：ScientificConsistency、EntityStructure、ScientificProcess。

输出格式（严格按照以下方式输出 JSON）
{
  "relevance": {
    "ScientificConsistency": { "score": 0 },
    "EntityStructure": { "score": 0 },
    "ScientificProcess": { "score": 0 }
  }
}
"""

# 路径配置
filtered_dir = "./scir_dataset/filtered_images"
metadata_path = "./scir_dataset/metadata_updated.json"
output_metadata_path = "./scir_dataset/classified_metadata.json"
cache_path = "classification_cache.json"

# ✅ 加载分类缓存
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding='utf-8') as f:
        classification_cache = json.load(f)
else:
    classification_cache = {}


def parse_json_response(response_text):
    import json, re

    # 去掉 markdown 包装
    response_text = response_text.strip()
    response_text = re.sub(r"^```(?:json)?", "", response_text)
    response_text = re.sub(r"```$", "", response_text)
    response_text = response_text.strip()

    # 尝试直接解析
    try:
        data = json.loads(response_text)
        # 如果是字典且包含 relevance，就返回
        if isinstance(data, dict) and "relevance" in data:
            return data
    except Exception:
        pass

    # 如果直接解析失败，尝试提取最外层完整 JSON
    # ✅ 改为贪婪匹配，取最大块
    matches = re.findall(r"\{[\s\S]*\}", response_text)
    if not matches:
        return get_default_result()

    # 优先尝试最长的 JSON 片段
    matches.sort(key=len, reverse=True)
    for m in matches:
        try:
            data = json.loads(m)
            if "relevance" in data:
                return data
            # 兼容嵌套层，如 {"result": {"relevance": {...}}}
            for v in data.values():
                if isinstance(v, dict) and "relevance" in v:
                    return v
        except Exception:
            continue

    # 全部失败则返回默认
    return get_default_result()

def get_default_result():
    return {
        "relevance": {
            "ScientificConsistency": {"score": 0},
            "EntityStructure": {"score": 0},
            "ScientificProcess": {"score": 0}
        },
        "confidence": 0
    }


def generate_labels(relevance_data):
    """根据评分规则生成标签（score >= 7）"""
    labels = []
    for dim in ["ScientificConsistency", "EntityStructure",  "ScientificProcess"]:
        dim_info = relevance_data.get(dim, {})
        # 安全检查：确保是字典且包含score
        if isinstance(dim_info, dict) and dim_info.get("score", 0) >= 7:
            labels.append(dim)
    print("DEBUG relevance_data:", json.dumps(relevance_data, indent=2, ensure_ascii=False))
    return labels


def classify_image(filename):
    """对单张图片进行多维度分类"""
    img_path = os.path.join(filtered_dir, filename)

    if filename in classification_cache:
        print(f"📋 从缓存读取: {filename}")
        cached_result = classification_cache[filename]

        # 从缓存中重建完整结果（包含动态生成的标签）
        result = {
            "relevance": cached_result["relevance"],
            "confidence": cached_result["confidence"],
            "labels": generate_labels(cached_result["relevance"])
        }
        return filename, result

    try:
        with open(img_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ 无法读取图像 {filename}: {e}")
        return filename, None

    messages = [
        {"role": "system", "content": classification_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请对这张科学图像进行多维度分类评分。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="internvl3.5-latest",
                messages=messages,
                temperature=0
            )

            reply = response.choices[0].message.content.strip()
            print(f"🔍 原始模型输出({filename}):\n{reply}")
            result = parse_json_response(reply)

            # 确保结果包含必要字段
            if "relevance" not in result:
                result["relevance"] = get_default_result()["relevance"]
            if "confidence" not in result:
                result["confidence"] = 0

            # 由代码生成标签（关键修改）
            labels = generate_labels(result["relevance"])

            # 保存到缓存（只存原始评分数据，不存标签）
            classification_cache[filename] = {
                "relevance": result["relevance"],
                "confidence": result["confidence"]
            }

            # 构建完整返回结果（仅用于内部处理，不写入元数据）
            full_result = {
                "relevance": result["relevance"],
                "confidence": result["confidence"],
                "labels": labels
            }

            print(f"✅ 分类完成: {filename} → Labels: {labels}")
            return filename, full_result

        except Exception as e:
            print(f"⚠️ 尝试 {attempt + 1}/{max_retries} 失败: {filename} - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"❌ 分类失败: {filename}")
                return filename, None


def main():
    print("=" * 60)
    print("🚀 开始科学图像多维度分类任务")
    print("=" * 60)

    # ✅ 1. 加载元数据
    print("\n📂 加载元数据文件...")
    try:
        with open(metadata_path, "r", encoding='utf-8') as f:
            metadata_list = json.load(f)
        print(f"✅ 成功加载 {len(metadata_list)} 条元数据记录")
    except Exception as e:
        print(f"❌ 无法加载元数据文件: {e}")
        return

    # 创建文件名到metadata的映射
    filename_to_metadata = {}
    for metadata in metadata_list:
        for segment in metadata.get("segments", []):
            filename = segment.get("filename")
            if filename:
                filename_to_metadata[filename] = (metadata, segment)

    # ✅ 2. 获取所有需要分类的图像
    print("\n📊 统计待分类图像...")
    image_files = []
    for fname in os.listdir(filtered_dir):
        if fname.startswith("sciir") and fname.endswith(".png"):
            image_files.append(fname)

    print(f"✅ 找到 {len(image_files)} 张待分类图像")

    # ✅ 3. 并行执行分类任务
    print("\n🔍 开始多线程分类处理...")
    classification_results = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有任务
        future_to_filename = {
            executor.submit(classify_image, fname): fname
            for fname in image_files
        }

        # 处理完成的任务
        completed = 0
        for future in as_completed(future_to_filename):
            filename, result = future.result()
            completed += 1

            if result:
                classification_results[filename] = result
                labels_str = ", ".join(result.get("labels", []))
                print(f"[{completed}/{len(image_files)}] ✅ {filename} → {labels_str if labels_str else '无标签'}")
            else:
                print(f"[{completed}/{len(image_files)}] ❌ {filename} → 分类失败")

    # ✅ 4. 更新元数据，只添加labels字段（关键修改）
    print("\n📝 更新元数据中的labels字段...")
    update_count = 0

    for metadata in metadata_list:
        for segment in metadata.get("segments", []):
            filename = segment.get("filename")
            if filename in classification_results:
                # 只添加labels字段，不添加任何relevance内容
                segment["labels"] = classification_results[filename].get("labels", [])
                update_count += 1

    # ✅ 5. 保存更新后的元数据
    print(f"\n💾 保存更新后的元数据...")
    try:
        with open(output_metadata_path, "w", encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"✅ 成功更新 {update_count} 个segment的labels字段")
        print(f"✅ 更新后的元数据已保存至: {output_metadata_path}")
    except Exception as e:
        print(f"❌ 保存元数据失败: {e}")

    # ✅ 6. 保存分类缓存
    print("\n💾 保存分类缓存...")
    try:
        with open(cache_path, "w", encoding='utf-8') as f:
            json.dump(classification_cache, f, indent=2, ensure_ascii=False)
        print(f"✅ 分类缓存已保存至: {cache_path}")
    except Exception as e:
        print(f"⚠️ 保存缓存失败: {e}")

    # ✅ 7. 统计分析
    print("\n📊 分类统计结果:")
    print("=" * 60)

    # 统计各维度标签分布
    label_counts = defaultdict(int)
    label_combinations = defaultdict(int)

    for result in classification_results.values():
        labels = result.get("labels", [])
        for label in labels:
            label_counts[label] += 1
        if labels:
            label_combinations[tuple(sorted(labels))] += 1

    print("\n各维度标签分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(classification_results)) * 100
        print(f"  {label}: {count} 张 ({percentage:.1f}%)")

    print("\n常见标签组合 (Top 10):")
    for combo, count in sorted(label_combinations.items(), key=lambda x: x[1], reverse=True)[:10]:
        combo_str = " + ".join(combo)
        percentage = (count / len(classification_results)) * 100
        print(f"  {combo_str}: {count} 张 ({percentage:.1f}%)")

    # 统计无标签图像
    no_label_count = sum(1 for r in classification_results.values() if not r.get("labels"))
    if no_label_count > 0:
        print(f"\n⚠️ 无标签图像: {no_label_count} 张 ({(no_label_count / len(classification_results)) * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("🎉 分类任务完成!")
    print(f"✅ 成功分类: {len(classification_results)}/{len(image_files)} 张图像")
    print(f"✅ 输出文件: {output_metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()