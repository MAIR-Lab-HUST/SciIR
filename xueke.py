import os
import json
import re
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from collections import defaultdict

# ======================================================
#        🚀 配置区域
# ======================================================

INPUT_JSON_PATH = "./scir_dataset/updated_metadata_3.json"
OUTPUT_JSON_PATH = "scir_dataset/classified_abstracts3.json"
CACHE_PATH = "abstract_classification_cache.json"

API_KEYS = [
    "sk-pmxwgcwdstzgjqvhvxaulhbnqkajhmvgleoxmjvioadgqdcg",
    "sk-bsrtizgzefklhyziwcoiifrrycdaagtrvupuyterfilcjkxw",
    "sk-fcffshebwviqxokknuuibustebyutizlveyakakdbspjkdox",
    "sk-gzpgdgjzbjvehsmvmkemydiyclhwjunmthevinrdyygejsiq"
]

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

# ======================================================
#        📚 学科映射字典 (核心修正逻辑)
# ======================================================

VALID_CATEGORIES = [
    "Physical sciences",
    "Earth and environmental sciences",
    "Biological sciences",
    "Health sciences",
    "Scientific community and society"
]

SUB_TO_MAIN_MAP = {
    # Physical sciences
    "Physics": "Physical sciences", "Astronomy": "Physical sciences", "Planetary science": "Physical sciences",
    "Chemistry": "Physical sciences", "Materials science": "Physical sciences", "Mathematics": "Physical sciences",
    "Computing": "Physical sciences", "Engineering": "Physical sciences", "Nanoscience": "Physical sciences",
    "Optics": "Physical sciences", "Photonics": "Physical sciences", "Energy science": "Physical sciences",

    # Earth and environmental sciences
    "Climate sciences": "Earth and environmental sciences", "Ecology": "Earth and environmental sciences",
    "Environmental sciences": "Earth and environmental sciences",
    "Solid Earth sciences": "Earth and environmental sciences",
    "Geology": "Earth and environmental sciences", "Ocean sciences": "Earth and environmental sciences",
    "Hydrology": "Earth and environmental sciences", "Natural hazards": "Earth and environmental sciences",
    "Limnology": "Earth and environmental sciences", "Space physics": "Earth and environmental sciences",

    # Biological sciences
    "Genetics": "Biological sciences", "Microbiology": "Biological sciences", "Neuroscience": "Biological sciences",
    "Immunology": "Biological sciences", "Evolution": "Biological sciences", "Cancer": "Biological sciences",
    "Cell biology": "Biological sciences", "Biochemistry": "Biological sciences",
    "Molecular biology": "Biological sciences",
    "Zoology": "Biological sciences", "Developmental biology": "Biological sciences",
    "Structural biology": "Biological sciences",
    "Physiology": "Biological sciences", "Bioinformatics": "Biological sciences",
    "Biotechnology": "Biological sciences",
    "Plant sciences": "Biological sciences", "Psychology": "Biological sciences", "Biophysics": "Biological sciences",

    # Health sciences
    "Diseases": "Health sciences", "Health care": "Health sciences", "Medical research": "Health sciences",
    "Anatomy": "Health sciences", "Pathogenesis": "Health sciences", "Biomarkers": "Health sciences",
    "Neurology": "Health sciences", "Endocrinology": "Health sciences", "Medicine": "Health sciences",

    # Scientific community and society
    "Scientific community": "Scientific community and society", "Social sciences": "Scientific community and society",
    "Business": "Scientific community and society", "Agriculture": "Scientific community and society",
    "Water resources": "Scientific community and society", "Geography": "Scientific community and society",
    "Forestry": "Scientific community and society"
}

# ======================================================
#        🛠️ 核心逻辑
# ======================================================

clients = [
    OpenAI(api_key=k, base_url="https://api.siliconflow.cn/v1")
    for k in API_KEYS
]
client_index = 0
client_lock = threading.Lock()
stop_event = threading.Event()


def get_client():
    global client_index
    with client_lock:
        client = clients[client_index]
        client_index = (client_index + 1) % len(clients)
    return client


SYSTEM_PROMPT = """
Role: Expert Scientific Taxonomist.
Task: Classify the abstract into EXACTLY ONE of the 5 Major Categories.

MAJOR CATEGORIES (Select ONE of these strictly):
1. Physical sciences
2. Earth and environmental sciences
3. Biological sciences
4. Health sciences
5. Scientific community and society

Instructions:
- Analyze the abstract and the provided subjects keywords.
- Output strictly a JSON object: {"category": "Your Selected Category"}
- ⛔ DO NOT output sub-disciplines. Output only the Major Category name.
"""

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding='utf-8') as f:
        classification_cache = json.load(f)
else:
    classification_cache = {}


def parse_json_response(response_text):
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text).strip()

    result = None
    try:
        data = json.loads(text)
        if "category" in data:
            result = data["category"]
    except json.JSONDecodeError:
        match = re.search(r'"category":\s*"([^"]+)"', text)
        if match:
            result = match.group(1)

    return result.strip() if result else None


def classify_abstract_task(item_id, text_content, subjects_info):
    """
    分类任务
    :param item_id: 图片ID
    :param text_content: 摘要文本
    :param subjects_info: 科目列表字符串 (e.g., "Molecular biology, Neuroscience")
    """
    if stop_event.is_set(): return item_id, None
    if item_id in classification_cache: return item_id, classification_cache[item_id]

    # 构建更丰富的 Prompt
    user_content = f"Abstract: {text_content}"
    if subjects_info:
        user_content += f"\n\nRelated Subjects/Keywords: {subjects_info}"

    # 如果两者都缺失，无法分类
    if (not text_content or len(text_content) < 5) and (not subjects_info or len(subjects_info) < 3):
        return item_id, "Unknown"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    max_retries = 5

    for attempt in range(max_retries):
        if stop_event.is_set(): return item_id, None

        client = get_client()

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                timeout=30.0
            )

            reply = response.choices[0].message.content.strip()
            category = parse_json_response(reply)

            final_category = None

            if category in VALID_CATEGORIES:
                final_category = category
            elif category in SUB_TO_MAIN_MAP:
                final_category = SUB_TO_MAIN_MAP[category]
                print(f"🔧 自动修正 {item_id}: {category} -> {final_category}")
            elif category:
                for vc in VALID_CATEGORIES:
                    if vc.lower() == category.lower():
                        final_category = vc
                        break

            if final_category:
                classification_cache[item_id] = final_category
                print(f"✅ {item_id} -> {final_category}")
                return item_id, final_category
            else:
                print(f"⚠️ 识别无效 ({attempt + 1}/{max_retries}): {item_id} 返回了 '{category}' -> 正在重试...")
                time.sleep(1)
                continue

        except Exception as e:
            error_msg = str(e)
            low_balance_keywords = ["insufficient", "balance", "余额不足", "30001"]
            if any(k in error_msg.lower() for k in low_balance_keywords):
                print(f"❗ 余额不足，停止任务: {error_msg}")
                stop_event.set()
                return item_id, None

            if "429" in error_msg or "rate limit" in error_msg.lower() or "tpm" in error_msg.lower():
                wait_time = 5 + (attempt * 2) + random.uniform(1, 4)
                print(f"⏳ TPM限流 ({item_id}): 等待 {wait_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue

            if attempt < max_retries - 1:
                print(f"⚠️ API错误重试 {attempt + 1}/{max_retries}: {error_msg}")
                time.sleep(2)
            else:
                print(f"❌ {item_id} 最终失败: {error_msg}")

    return item_id, None


# ======================================================
#        ✨ 标签替换逻辑 (Consistency -> Law)
# ======================================================
def process_label_replacement(data_list):
    """
    遍历 data_list 中的所有 item -> segments -> labels
    如果有 "ScientificConsistency"，将其替换为 "ScientificLaw"。
    """
    print("\n🔄 开始执行标签替换 (ScientificConsistency -> ScientificLaw)...")
    replacement_count = 0
    segment_count = 0

    for item in data_list:
        segments = item.get("segments", [])
        if not segments or not isinstance(segments, list):
            continue

        for seg in segments:
            labels = seg.get("labels", [])
            if not labels or not isinstance(labels, list):
                continue

            if "ScientificConsistency" in labels:
                new_labels = [
                    "ScientificLaw" if label == "ScientificConsistency" else label
                    for label in labels
                ]
                seg["labels"] = new_labels
                replacement_count += 1
            segment_count += 1

    print(f"✅ 标签替换完成: 检查了 {segment_count} 个 Segment，修正了 {replacement_count} 处标签。")


def main():
    print("=" * 60)
    print("🚀 科学论文摘要分类 + Subject增强 + 标签修正")
    print("=" * 60)

    try:
        with open(INPUT_JSON_PATH, "r", encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"📂 加载了 {len(data_list)} 条数据")
    except Exception as e:
        print(f"❌ 读取输入文件失败: {e}")
        return

    # 1. 准备分类任务，同时提取 subjects
    tasks = []
    for item in data_list:
        img_id = item.get("image_id")
        abstract = item.get("article_abstract", "")

        # 提取 subjects 并转为字符串
        subjects_list = item.get("subjects", [])
        subjects_str = ""
        if isinstance(subjects_list, list):
            subjects_str = ", ".join([str(s) for s in subjects_list])

        # 如果摘要为空，使用标题
        if not abstract or len(abstract.strip()) < 10:
            abstract = item.get("article_title", "")

        tasks.append((img_id, abstract, subjects_str))

    results_map = {}
    num_workers = 6

    print(f"🔥 启动 {num_workers} 个线程处理分类任务...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 注意这里传参增加了 subjects_str
        future_to_id = {
            executor.submit(classify_abstract_task, tid, content, subj): tid
            for tid, content, subj in tasks
        }

        completed = 0
        try:
            for future in as_completed(future_to_id):
                if stop_event.is_set():
                    print("🛑 全局停止触发...")
                    break

                img_id, category = future.result()
                completed += 1
                if category:
                    results_map[img_id] = category

                if completed % 20 == 0:
                    print(f"进度: {completed}/{len(tasks)}")

        except KeyboardInterrupt:
            stop_event.set()
            print("\n🛑 用户强制停止")

    # 2. 更新分类结果到 data_list
    print("\n📥 更新分类结果到内存...")
    update_count = 0
    for item in data_list:
        img_id = item.get("image_id")
        if img_id in results_map:
            item["subject_category"] = results_map[img_id]
            update_count += 1
        elif img_id in classification_cache:
            item["subject_category"] = classification_cache[img_id]
            update_count += 1
        else:
            if "subject_category" not in item:
                item["subject_category"] = None

    print(f"✅ 分类更新完毕，成功分类: {update_count}/{len(data_list)}")

    # 3. 执行标签替换
    process_label_replacement(data_list)

    # 4. 保存最终结果
    print(f"\n💾 保存最终结果到 {OUTPUT_JSON_PATH} ...")
    with open(OUTPUT_JSON_PATH, "w", encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)

    # 保存缓存
    with open(CACHE_PATH, "w", encoding='utf-8') as f:
        json.dump(classification_cache, f, indent=2, ensure_ascii=False)

    print("🎉 所有任务全部完成！")


if __name__ == "__main__":
    main()