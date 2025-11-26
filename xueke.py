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

INPUT_JSON_PATH = "./scir_dataset/updated_metadata_6.json"
OUTPUT_JSON_PATH = "./scir_dataset/classified_abstracts.json"
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

# 如果模型不听话返回了子学科，我们在这里做一个兜底映射，避免浪费重试次数
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


# 强化Prompt，明确要求只输出大类
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
- Analyze the abstract content based on standard scientific taxonomy.
- Output strictly a JSON object: {"category": "Your Selected Category"}
- ⛔ DO NOT output sub-disciplines (e.g., do NOT output "Chemistry" or "Genetics"). Output only the Major Category name.
"""

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding='utf-8') as f:
        classification_cache = json.load(f)
else:
    classification_cache = {}


def parse_json_response(response_text):
    text = response_text.strip()
    # 清理 markdown
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


def classify_abstract_task(item_id, text_content):
    """包含严格校验和重试机制的分类任务"""
    if stop_event.is_set(): return item_id, None
    if item_id in classification_cache: return item_id, classification_cache[item_id]

    # 内容过短处理
    if not text_content or len(text_content) < 5:
        return item_id, "Unknown"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Abstract: {text_content}"}
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

            # =========================================
            # 🛡️ 严格校验与修正逻辑 (Strict Check)
            # =========================================

            final_category = None

            # 情况1: 直接命中大类 -> 完美
            if category in VALID_CATEGORIES:
                final_category = category

            # 情况2: 命中子学科 -> 自动修正 (避免重试)
            elif category in SUB_TO_MAIN_MAP:
                final_category = SUB_TO_MAIN_MAP[category]
                print(f"🔧 自动修正 {item_id}: {category} -> {final_category}")

            # 情况3: 模糊匹配 (最后的挣扎)
            elif category:
                for vc in VALID_CATEGORIES:
                    if vc.lower() == category.lower():
                        final_category = vc
                        break

            # 判定结果
            if final_category:
                classification_cache[item_id] = final_category
                print(f"✅ {item_id} -> {final_category}")
                return item_id, final_category
            else:
                # ❌ 如果这里仍然没有有效结果，说明模型输出了不在列表里的东西
                print(f"⚠️ 识别无效 ({attempt + 1}/{max_retries}): {item_id} 返回了 '{category}' -> 正在重试...")
                # 这里不return，直接进入下一次循环，相当于触发重试
                time.sleep(1)  # 稍作停顿
                continue

        except Exception as e:
            error_msg = str(e)

            # 欠费检测
            low_balance_keywords = ["insufficient", "balance", "余额不足", "30001"]
            if any(k in error_msg.lower() for k in low_balance_keywords):
                print(f"❗ 余额不足，停止任务: {error_msg}")
                stop_event.set()
                return item_id, None

            # 429 限流检测
            if "429" in error_msg or "rate limit" in error_msg.lower() or "tpm" in error_msg.lower():
                wait_time = 5 + (attempt * 2) + random.uniform(1, 4)
                print(f"⏳ TPM限流 ({item_id}): 等待 {wait_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue

            # 其他错误
            if attempt < max_retries - 1:
                print(f"⚠️ API错误重试 {attempt + 1}/{max_retries}: {error_msg}")
                time.sleep(2)
            else:
                print(f"❌ {item_id} 最终失败: {error_msg}")

    return item_id, None


def main():
    print("=" * 60)
    print("🚀 科学论文摘要分类 (含子学科自动修正 + 严格重试)")
    print("=" * 60)

    try:
        with open(INPUT_JSON_PATH, "r", encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"📂 加载了 {len(data_list)} 条数据")
    except Exception as e:
        print(f"❌ 读取输入文件失败: {e}")
        return

    tasks = []
    for item in data_list:
        img_id = item.get("image_id")
        abstract = item.get("article_abstract", "")
        # 如果摘要为空，使用标题
        if not abstract or len(abstract.strip()) < 10:
            abstract = item.get("article_title", "")
        tasks.append((img_id, abstract))

    results_map = {}
    num_workers = 6

    print(f"🔥 启动 {num_workers} 个线程处理任务...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_id = {
            executor.submit(classify_abstract_task, tid, content): tid
            for tid, content in tasks
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

    # 更新与保存
    print("\n💾 保存结果...")
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

    with open(OUTPUT_JSON_PATH, "w", encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)

    with open(CACHE_PATH, "w", encoding='utf-8') as f:
        json.dump(classification_cache, f, indent=2, ensure_ascii=False)

    print(f"✅ 完成！成功分类: {update_count}/{len(data_list)}")
    print(f"📄 文件: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()