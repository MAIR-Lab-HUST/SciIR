import os
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from collections import defaultdict
import time
import threading  # ✅ 1. 导入 threading

# ======================================================
#        🚀 多 API KEY 加速版本
# ======================================================
API_KEYS = [
    "sk-izrxeahvbxtrpjvujgdqmrabzlhpcngbtwposuwoiugskhwj",
    "sk-vlvawvmsujhyexstyfccwnodxcfgtgbzdirityejqnlyncat"
]

# 创建多个客户端
clients = [
    OpenAI(api_key=k, base_url="https://api.siliconflow.cn/v1")
    for k in API_KEYS
]

client_index = 0
client_lock = threading.Lock()

# --- ADDED/CHANGED: 全局停止事件，当检测到欠费或其它致命错误时设置该事件，通知主线程终止 ---
stop_event = threading.Event()
# --------------------------------------------------------------------------------------------

def get_client():
    """轮询分配 API KEY（线程安全）"""
    global client_index
    with client_lock:
        client = clients[client_index]
        client_index = (client_index + 1) % len(clients)
    return client

# ========== 保持原有的 classification_prompt 等不变 ==========

classification_prompt = """
角色与任务
你是一个严谨的科学图像分类器。你的任务是围绕下述四个维度，对输入图像进行多标签、多维度的相关度评分（1–10）。
...
"""  # 截略显示，实际保持原内容

# 路径配置
filtered_dir = "./scir_dataset/filtered_images_3"
metadata_path = "./scir_dataset/updated_metadata_3.json"
output_metadata_path = "./scir_dataset/classified_metadata3.json"
cache_path = "classification_cache.json"

# ✅ 加载分类缓存
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding='utf-8') as f:
        classification_cache = json.load(f)
else:
    classification_cache = {}

# ... 保留 parse_json_response, get_default_result, generate_labels 等函数不变 ...

def parse_json_response(response_text):
    import json, re
    response_text = response_text.strip()
    response_text = re.sub(r"^```(?:json)?", "", response_text)
    response_text = re.sub(r"```$", "", response_text)
    response_text = response_text.strip()
    try:
        data = json.loads(response_text)
        if isinstance(data, dict) and "relevance" in data:
            return data
    except Exception:
        pass
    matches = re.findall(r"\{[\s\S]*\}", response_text)
    if not matches:
        return get_default_result()
    matches.sort(key=len, reverse=True)
    for m in matches:
        try:
            data = json.loads(m)
            if "relevance" in data:
                return data
            for v in data.values():
                if isinstance(v, dict) and "relevance" in v:
                    return v
        except Exception:
            continue
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
    labels = []
    for dim in ["ScientificConsistency", "EntityStructure", "ScientificProcess"]:
        dim_info = relevance_data.get(dim, {})
        if isinstance(dim_info, dict) and dim_info.get("score", 0) >= 7:
            labels.append(dim)
    print("DEBUG relevance_data:", json.dumps(relevance_data, indent=2, ensure_ascii=False))
    return labels

# ========== 这里是对 classify_image 的关键修改（detect 欠费即 stop） ==========
def classify_image(filename):
    """对单张图片进行多维度分类"""
    img_path = os.path.join(filtered_dir, filename)

    # 如果全局停止事件已触发，直接跳过并返回 None
    if stop_event.is_set():
        print(f"⛔ 全局停止已触发，跳过: {filename}")
        return filename, None

    if filename in classification_cache:
        print(f"📋 从缓存读取: {filename}")
        cached_result = classification_cache[filename]
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

    max_retries = 5
    for attempt in range(max_retries):

        # 在每次尝试前检查全局停止信号
        if stop_event.is_set():
            print(f"⛔ 全局停止已触发，终止对 {filename} 的尝试")
            return filename, None

        client = get_client()

        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-VL-30B-A3B-Instruct",
                messages=messages,
                temperature=0,
                timeout=30.0
            )

            reply = response.choices[0].message.content.strip()
            print(f"🔍 原始模型输出({filename}):\n{reply}")
            result = parse_json_response(reply)

            if "relevance" not in result:
                result["relevance"] = get_default_result()["relevance"]
            if "confidence" not in result:
                result["confidence"] = 0

            labels = generate_labels(result["relevance"])

            classification_cache[filename] = {
                "relevance": result["relevance"],
                "confidence": result["confidence"]
            }

            full_result = {
                "relevance": result["relevance"],
                "confidence": result["confidence"],
                "labels": labels
            }

            print(f"✅ 分类完成: {filename} → Labels: {labels}")
            return filename, full_result

        except Exception as e:
            # --- ADDED: 检测是否为欠费错误（code 30001 或 提示 'balance' / 'insufficient'） ---
            error_msg = str(e)
            print(f"⚠️ 尝试 {attempt + 1}/{max_retries} 失败: {filename} - {error_msg}")

            # 尝试提取 code 字段
            code_match = re.search(r"""['"]?code['"]?\s*[:=]\s*([0-9]+)""", error_msg)
            detected_code = int(code_match.group(1)) if code_match else None

            # 另外检测关键词 'insufficient' / 'balance'
            low_balance_keywords = ["insufficient", "balance", "余额不足", "30001"]

            if detected_code == 30001 or any(k in error_msg.lower() for k in low_balance_keywords):
                # 识别为欠费或余额不足 -> 触发全局停止
                print(f"❗ 检测到账号余额不足或错误码 30001（{detected_code}），触发全局停止。")
                stop_event.set()
                # 直接返回失败，主线程会检测到 stop_event 并取消其他任务
                return filename, None

            # 原有的重试/退避逻辑
            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                print(f"⏳ API限流... 稍后重试")
                time.sleep(2 * (attempt + 1))
            elif "timeout" in error_msg.lower():
                print(f"⏱️ 请求超时... 稍后重试")
                time.sleep(3)
            else:
                if attempt < max_retries - 1:
                    time.sleep(2)

            if attempt == max_retries - 1:
                print(f"❌ 分类失败: {filename}")
                return filename, None
            # 循环继续重试（除非 stop_event 被触发）
# ============================================================================

def main():
    print("=" * 60)
    print("🚀 开始科学图像多维度分类任务 (使用 SiliconFlow 多API)")
    print("=" * 60)

    print("\n📂 加载元数据文件...")
    try:
        with open(metadata_path, "r", encoding='utf-8') as f:
            metadata_list = json.load(f)
        print(f"✅ 成功加载 {len(metadata_list)} 条元数据记录")
    except Exception as e:
        print(f"❌ 无法加载元数据文件: {e}")
        return

    filename_to_metadata = {}
    for metadata in metadata_list:
        for segment in metadata.get("segments", []):
            filename = segment.get("filename")
            if filename:
                filename_to_metadata[filename] = (metadata, segment)

    print("\n📊 统计待分类图像...")
    image_files = []
    for fname in os.listdir(filtered_dir):
        if fname.startswith("sciir") and fname.endswith(".png"):
            image_files.append(fname)

    print(f"✅ 找到 {len(image_files)} 张待分类图像")

    print("\n🔍 开始多线程分类处理...")
    classification_results = {}

    num_workers = 4
    print(f"ℹ️ 使用 {num_workers} 个工作线程 (API Key 数量: {len(API_KEYS)})")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_filename = {
            executor.submit(classify_image, fname): fname
            for fname in image_files
        }

        completed = 0
        try:
            for future in as_completed(future_to_filename):
                # 一旦检测到全局停止信号，尝试取消其余任务并跳出
                if stop_event.is_set():
                    print("❗ 全局停止信号已触发，正在取消剩余任务...")
                    # 尝试取消还未完成的 future（注意：已提交任务可能无法被取消）
                    for fut in future_to_filename:
                        if not fut.done():
                            fut.cancel()
                    break

                filename, result = future.result()
                completed += 1

                if result:
                    classification_results[filename] = result
                    labels_str = ", ".join(result.get("labels", []))
                    print(f"[{completed}/{len(image_files)}] ✅ {filename} → {labels_str if labels_str else '无标签'}")
                else:
                    print(f"[{completed}/{len(image_files)}] ❌ {filename} → 分类失败")

        except KeyboardInterrupt:
            print("🛑 收到 KeyboardInterrupt，准备停止...")
            stop_event.set()
            # 尝试取消所有未完成的任务
            for fut in future_to_filename:
                if not fut.done():
                    fut.cancel()

    # 如果是因为 stop_event 导致中断，告知原因
    if stop_event.is_set():
        print("\n❗ 运行被中止（可能原因：账户余额不足或检测到致命错误）。")
        # 这里可以选择是否保存当前缓存 / 元数据 —— 我们继续保存缓存与已完成的结果
    # ========== 下面保持你原来的元数据更新、保存缓存与统计逻辑 ==========

    print("\n📝 更新元数据中的labels字段...")
    update_count = 0

    for metadata in metadata_list:
        for segment in metadata.get("segments", []):
            filename = segment.get("filename")
            if filename in classification_results:
                segment["labels"] = classification_results[filename].get("labels", [])
                update_count += 1

    print(f"\n💾 保存更新后的元数据...")
    try:
        with open(output_metadata_path, "w", encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"✅ 成功更新 {update_count} 个segment的labels字段")
        print(f"✅ 更新后的元数据已保存至: {output_metadata_path}")
    except Exception as e:
        print(f"❌ 保存元数据失败: {e}")

    print("\n💾 保存分类缓存...")
    try:
        with open(cache_path, "w", encoding='utf-8') as f:
            json.dump(classification_cache, f, indent=2, ensure_ascii=False)
        print(f"✅ 分类缓存已保存至: {cache_path}")
    except Exception as e:
        print(f"⚠️ 保存缓存失败: {e}")

    # 统计输出（保持不变）
    print("\n📊 分类统计结果:")
    print("=" * 60)

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
        percentage = (count / len(classification_results)) * 100 if classification_results else 0
        print(f"  {label}: {count} 张 ({percentage:.1f}%)")

    print("\n常见标签组合 (Top 10):")
    for combo, count in sorted(label_combinations.items(), key=lambda x: x[1], reverse=True)[:10]:
        combo_str = " + ".join(combo)
        percentage = (count / len(classification_results)) * 100 if classification_results else 0
        print(f"  {combo_str}: {count} 张 ({percentage:.1f}%)")

    no_label_count = sum(1 for r in classification_results.values() if not r.get("labels"))
    if no_label_count > 0 and classification_results:
        print(f"\n⚠️ 无标签图像: {no_label_count} 张 ({(no_label_count / len(classification_results)) * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("🎉 分类任务结束（或已被中止）!")
    print(f"✅ 成功分类: {len(classification_results)}/{len(image_files)} 张图像")
    print(f"✅ 输出文件: {output_metadata_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
