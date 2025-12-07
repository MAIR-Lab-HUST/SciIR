import json
import os
import shutil
import numpy as np

# ================= 配置路径 =================
ABSTRACTS_PATH = "scir_dataset/classified_abstracts3.json"
CAPTIONS_PATH = "scir_dataset/caption3.json"
OUTPUT_ROOT = "output_dataset3"  # 输出根目录
IMAGES_SOURCE_DIR = "scir_dataset/filtered_images_3"  # 源图片目录


# ================= 辅助函数 =================

def normalize_path(p):
    """标准化路径，提取文件名作为唯一标识符"""
    if not p:
        return ""
    return os.path.basename(p)


def count_terms(reasoning):
    """计算 reasoning 中 terms 的总数"""
    if not reasoning:
        return 0
    total = 0
    keys = ["ScientificLaw", "EntityStructure", "ScientificProcess"]
    for k in keys:
        if reasoning.get(k) and isinstance(reasoning[k], dict) and "terms" in reasoning[k]:
            terms = reasoning[k]["terms"]
            if terms:
                total += len(terms)
    return total


def clean_reasoning(reasoning):
    """
    清理 reasoning: 如果某个 label 的 terms 和 visualization 都是空的，
    将该 label 的值设为 null。用于最终输出的美观和规范。
    """
    if not reasoning:
        return reasoning

    cleaned = reasoning.copy()
    keys = ["ScientificLaw", "EntityStructure", "ScientificProcess"]

    for k in keys:
        if k in cleaned and cleaned[k]:
            label_data = cleaned[k]
            # 检查 terms 是否为空
            terms = label_data.get("terms", [])
            terms_empty = not terms or len(terms) == 0

            # 检查 visualization 是否为空
            visualization = label_data.get("visualization", [])
            viz_empty = not visualization or len(visualization) == 0

            # 如果都是空的，设为 null
            if terms_empty and viz_empty:
                cleaned[k] = None

    return cleaned


def load_json(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ================= 主逻辑 =================

def get_quartiles(data):
    """计算数据的上下四分位数 (Q1, Q3)"""
    if not data:
        return 0, 0
    return np.percentile(data, 25), np.percentile(data, 75)


def filter_by_iqr(items, key_func):
    """
    根据 key_func 提取的值，保留位于 [Q1, Q3] 之间的项。
    """
    if not items:
        return []

    values = [key_func(item) for item in items]
    q1, q3 = get_quartiles(values)

    filtered = []
    for item in items:
        val = key_func(item)
        if q1 <= val <= q3:
            filtered.append(item)
    return filtered


def main():
    print("1. Loading datasets...")
    abstracts_data = load_json(ABSTRACTS_PATH)
    captions_data = load_json(CAPTIONS_PATH)

    if not abstracts_data or not captions_data:
        print("Data load failed. Exiting.")
        return

    # 构建 Caption 索引
    caption_map = {normalize_path(item['image_path']): item for item in captions_data}

    # 预定义关注的 Label 集合
    VALID_LABELS = ["ScientificLaw", "EntityStructure", "ScientificProcess"]

    # 初始化四个分组
    groups = {
        "EntityStructure_ScientificLaw": [],
        "ScientificLaw_ScientificProcess": [],
        "EntityStructure_ScientificProcess": [],
        "All_Three": []
    }

    print("2. Grouping candidates by checking reasoning content...")
    processed_files = set()  # 防止重复处理同一张图片

    for article in abstracts_data:
        for segment in article.get("segments", []):
            img_filename = segment.get("filename")

            # 尝试通过 filename 或 path 找到对应的 caption
            caption_item = caption_map.get(img_filename)
            if not caption_item:
                caption_item = caption_map.get(normalize_path(segment.get("path", "")))

            if not caption_item:
                continue

            # 去重检查 (有些 abstract 可能引用同一张图)
            if img_filename in processed_files:
                continue
            processed_files.add(img_filename)

            # ================= [核心修改] =================
            # 不再使用 segment['labels']，而是检查 reasoning 中的 terms 是否真实存在
            reasoning = caption_item.get("reasoning", {})
            current_labels = []

            for label in VALID_LABELS:
                label_data = reasoning.get(label)
                # 只有当 label 数据存在，且 terms 列表非空时，才认为该标签有效
                if label_data and isinstance(label_data, dict):
                    terms = label_data.get("terms", [])
                    if terms and len(terms) > 0:
                        current_labels.append(label)

            # 排序以确保 key 组合顺序一致
            current_labels.sort()
            # ================= [修改结束] =================

            group_name = None
            if len(current_labels) == 2:
                group_name = f"{current_labels[0]}_{current_labels[1]}"
            elif len(current_labels) == 3:
                group_name = "All_Three"

            # 如果不属于这4组，跳过
            if not group_name or group_name not in groups:
                continue

            # 提取所需字段
            rendered_txt = caption_item.get("rendered_text_stage2", [])
            retained_txt = caption_item.get("retained_text_stage3", [])

            # 计算 total terms 用于后续筛选
            term_count = count_terms(reasoning)

            # 清理 reasoning (仅用于输出显示，将全空的设为null)
            cleaned_reasoning = clean_reasoning(reasoning)

            # 构建对象
            merged_obj = {
                "source_image_id": article.get("image_id"),
                "image_filename": img_filename,
                "original_path": segment.get("path"),
                "labels": current_labels,  # 这里使用的是基于内容判断出的真实 label
                "reasoning": cleaned_reasoning,
                "term_count": term_count,
                "rendered_len": len(rendered_txt),
                "retained_len": len(retained_txt),
                "sci-RCoT": caption_item.get("sci-RCoT"),
                "science_abstract_prompt": caption_item.get("science_abstract_prompt"),
                "retained_text": retained_txt
            }

            groups[group_name].append(merged_obj)

    # 处理每个分组
    print("3. Processing groups (Filtering & Splitting)...")

    for group_name, candidates in groups.items():
        print(f"\n--- Processing Group: {group_name} (Initial: {len(candidates)}) ---")
        if not candidates:
            continue

        # --- 筛选步骤 1: term_count 位于 [Q1, Q3] ---
        candidates_step1 = filter_by_iqr(candidates, lambda x: x['term_count'])
        print(f"   After Term Count Filter: {len(candidates_step1)}")

        if not candidates_step1:
            continue

        # --- 筛选步骤 2: rendered_text 和 retained_text 数量均位于 [Q1, Q3] ---
        # 基于 Step 1 结果分布进行筛选
        candidates_step2a = filter_by_iqr(candidates_step1, lambda x: x['rendered_len'])
        candidates_step2b = filter_by_iqr(candidates_step2a, lambda x: x['retained_len'])

        final_candidates = candidates_step2b
        print(f"   After Text Length Filters: {len(final_candidates)}")

        if not final_candidates:
            continue

        # --- 分级步骤: Prompt vs CoT (Median Split) ---
        # 基于最终筛选出的样本的中位数
        term_counts = [x['term_count'] for x in final_candidates]
        median_val = np.median(term_counts)
        print(f"   Median Term Count (for split): {median_val:.2f}")

        prompt_ds = []
        cot_ds = []
        for item in final_candidates:
            # 清理辅助统计字段，准备保存
            out_item = item.copy()
            del out_item['rendered_len']
            del out_item['retained_len']
            del out_item['term_count']

            if item['term_count'] < median_val:
                prompt_ds.append(out_item)
            else:
                cot_ds.append(out_item)

        print(f"   > Prompt: {len(prompt_ds)}, CoT: {len(cot_ds)}")

        # --- 保存 ---
        sub_tasks = [("prompt", prompt_ds), ("CoT", cot_ds)]

        for sub_name, ds in sub_tasks:
            # 如果结果为空，也可能不需要建立文件夹，视需求而定。这里选择建立。
            if not ds:
                continue

            # 路径构建: OUTPUT_ROOT / GroupName / prompt_dataset / ...
            ds_folder_name = f"{sub_name}_dataset"
            base_dir = os.path.join(OUTPUT_ROOT, group_name, ds_folder_name)
            img_dir = os.path.join(base_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

            # 保存 JSON
            save_json(ds, os.path.join(base_dir, f"{sub_name}_data.json"))

            # 复制图片
            for item in ds:
                img_filename = item['image_filename']
                if img_filename:
                    # 从配置的源图片目录构建路径
                    src_path = os.path.join(IMAGES_SOURCE_DIR, img_filename)
                    if os.path.exists(src_path):
                        shutil.copy(src_path, os.path.join(img_dir, img_filename))
                    else:
                        print(f"   Warning: Image not found: {src_path}")

    print(f"\nDone! All outputs generated in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()