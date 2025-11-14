import os
import base64
import shutil
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from collections import defaultdict
import threading
import time
import signal

# ======================================================
#        🚀 多 API KEY 加速版本（直接填多个 key）
# ======================================================
API_KEYS = [
    "sk-lleqfnxceutjvaywjtruxsfrydpidbsxcpmjfinmwtbdzipf",
    "sk-hdbmdzhalqlzpsfgkjjxvohysihlzddfjgxwzronfxihbndo",
    # "sk-pwaczhradjqycscbzvomfephetmlliriddmdjvxgjwnoxmfx",
    # "sk-nhvkwjwwdzryadpxoyanovuuwkedndprviaaekbgpgdrgalq",
    # "sk-izrxeahvbxtrpjvujgdqmrabzlhpcngbtwposuwoiugskhwj",
    # "sk-vlvawvmsujhyexstyfccwnodxcfgtgbzdirityejqnlyncat"
]

# 创建多个客户端
clients = [
    OpenAI(api_key=k, base_url="https://api.siliconflow.cn/v1")
    for k in API_KEYS
]

client_index = 0
client_lock = threading.Lock()


def get_client():
    """轮询分配 API KEY（线程安全）"""
    global client_index
    with client_lock:
        client = clients[client_index]
        client_index = (client_index + 1) % len(clients)
    return client


# ======================================================
#        原程序配置保持不变
# ======================================================
system_prompt = """
你是一个图像识别助手。你接下来将看到一些从科学论文中分割出来的面板图像。
请判断该图像是否属于"科学图像"。  
"科学图像"指的是排除以下类别外、内容抽象且可绘制的科研示意图。  
排除以下情况：
- 图像不完整、存在多个相互独立的面板或有内嵌小图
- 任何包含刻度、单位、坐标轴等的数据可视化/统计图（包括但不限于：弦图、折线图、散点图、柱状图、箱线图、波形图、热力图、三维曲面图、等高线图、雷达图、瀑布图、系统发育树）
- 含有任何地图、显微镜图、组织切片、生物样本、骨骼轮廓图、分子结构图、器官解剖图、晶体结构图等用于展示真实世界内容或实验样本的图片 
- 由专业软件生成的渲染/仿真/三维重建图（含伪彩色渲染效果）
- 只含有文字、字母、符号或图注，或者纯文字流程图、纯图例、简单图标
输出要求：  
只输出以下两种结果之一：  
- "科学图像"  
- "不符合"
请严格按照定义判断，不做额外解释或描述
"""

input_dir = "./scir_dataset/filled_images_3"
filtered_dir = "./scir_dataset/filtered_images_silcon_31"
metadata_path = "./scir_dataset/cropped_metadata_3.json"
os.makedirs(filtered_dir, exist_ok=True)

# ======================================================
#        ✅ 修复版缓存管理（核心修复）
# ======================================================
cache_path = "filter_cache.json"
cache_lock = threading.Lock()


def load_safe_cache():
    """安全加载缓存，自动处理损坏文件"""
    if not os.path.exists(cache_path):
        print("📁 创建新缓存文件")
        return {}

    try:
        with open(cache_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        print(f"📊 成功加载缓存: {len(data)}条记录")
        return data
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        print(f"⚠️ 缓存文件损坏: {str(e)[:50]}... | 尝试恢复")
        # 备份损坏文件
        backup = f"{cache_path}.bak.{int(time.time())}"
        try:
            shutil.copy2(cache_path, backup)
            print(f"✅ 损坏缓存已备份至: {backup}")
        except Exception as backup_err:
            print(f"❌ 备份失败: {backup_err}")

        # 创建新缓存
        print("🔄 创建新的空缓存")
        return {}
    except Exception as e:
        print(f"❌ 未知错误: {e} | 创建新缓存")
        return {}


def save_cache_atomic():
    """原子写入缓存，防止文件损坏"""
    with cache_lock:
        temp_path = f"{cache_path}.tmp"
        try:
            # 先写入临时文件
            with open(temp_path, "w", encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

            # 原子替换（关键！）
            os.replace(temp_path, cache_path)
            print(f"💾 安全保存缓存: {len(cache)}条记录")
        except Exception as e:
            print(f"❗ 缓存保存失败: {e}")
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)


# 初始化缓存（安全加载）
cache = load_safe_cache()


# ======================================================
#                     工具函数
# ======================================================
def parse_filename(filename):
    match = re.match(r"sciir_img_(\d{6})_(\d{2})\.png", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


# ======================================================
#    🔥 核心：使用多 API KEY 并发调用的图像识别函数
# ======================================================
def process_image(filename):
    """多 API Key 版本的并行识别"""

    img_path = os.path.join(input_dir, filename)

    # 缓存命中
    if filename in cache:
        print(f"⚡ 缓存命中: {filename} → {cache[filename]}")
        return filename, cache[filename]

    # 编码图片
    try:
        with open(img_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ 读取图片失败 {filename}: {e}")
        return filename, "❌ Failed"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}},
            {"type": "text", "text": "这张图像是否为'科学图像'？"}
        ]}
    ]

    max_retries = 4
    for attempt in range(max_retries):

        client = get_client()  # ⬅ 使用不同 API KEY

        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct",
                messages=messages,
                temperature=0,
                timeout=30.0  # 增加超时控制
            )
            reply = response.choices[0].message.content.strip()

            # 更新内存缓存（不再直接写文件）
            with cache_lock:
                cache[filename] = reply

            return filename, reply

        except Exception as e:
            error_msg = str(e).lower()
            # 处理特定错误
            if "rate limit" in error_msg or "quota" in error_msg:
                print(f"⏳ API限流: {filename} | 重试 {attempt + 1}/{max_retries}")
                time.sleep(2 * (attempt + 1))  # 指数退避
            elif "timeout" in error_msg:
                print(f"⏱️ 请求超时: {filename} | 重试 {attempt + 1}/{max_retries}")
            else:
                print(f"⚠️ API错误: {filename} | {e} | 重试 {attempt + 1}/{max_retries}")

            time.sleep(1)
            continue

    # 所有重试失败
    with cache_lock:
        cache[filename] = "❌ Failed"
    return filename, "❌ Failed"


# ======================================================
#               1. 加载 metadata
# ======================================================
print("\n" + "=" * 50)
print("🔍 加载元数据文件...")
segment_info = {}
metadata_list = []

if os.path.exists(metadata_path):
    try:
        with open(metadata_path, "r", encoding='utf-8') as f:
            metadata_list = json.load(f)

        for metadata in metadata_list:
            image_id = metadata["image_id"]
            for seg in metadata["segments"]:
                segment_info[seg["filename"]] = (seg["width"], seg["height"], image_id)

        print(f"✅ 加载 {len(metadata_list)} 条元数据，{len(segment_info)} 个 segment")
    except Exception as e:
        print(f"❌ 加载metadata失败: {e}")
else:
    print("⚠️ metadata文件不存在，将跳过关联处理")
print("=" * 50 + "\n")

# ======================================================
#          2. 并发处理图像（已绑定多 API KEY）
# ======================================================
print("\n📋 处理模式: 处理所有文件")

all_results = []
valid_files = []  # (original_id, filename, width, height)

# 设置优雅退出信号
stop_event = threading.Event()


def signal_handler(sig, frame):
    print("\n\n🛑 检测到中断信号 (Ctrl+C)...")
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)

with ThreadPoolExecutor(max_workers=30) as executor:
    tasks = {}
    skipped = 0

    # 收集任务 - 处理所有 sciir 开头的 png 文件
    for fname in os.listdir(input_dir):
        if stop_event.is_set():
            break

        if fname.startswith("sciir") and fname.endswith(".png"):
            # 直接添加所有符合条件的文件
            tasks[executor.submit(process_image, fname)] = fname
        else:
            skipped += 1

    print(f"📊 需要处理: {len(tasks)} 张图，跳过: {skipped} (非 sciir 文件)")
    print(f"⏳ 开始处理 (按 Ctrl+C 安全中断)...")

    processed_count = 0
    last_save = 0
    SAVE_INTERVAL = 100  # 每100张保存一次缓存

    for future in as_completed(tasks):
        if stop_event.is_set():
            # 取消所有未完成的任务
            for f in tasks:
                if not f.done():
                    f.cancel()
            break

        filename = tasks[future]
        try:
            filename, reply = future.result(timeout=60)  # 增加单任务超时
        except Exception as e:
            print(f"🔥 任务异常 {filename}: {e}")
            filename, reply = filename, "❌ Failed"

        all_results.append((filename, reply))
        processed_count += 1

        # 定期保存缓存
        if processed_count - last_save >= SAVE_INTERVAL:
            print(f"\n🔄 [定期保存] 已处理 {processed_count}/{len(tasks)} 张图")
            save_cache_atomic()
            last_save = processed_count

        if reply.strip() == "科学图像":
            original_id, _ = parse_filename(filename)
            width, height = None, None
            if filename in segment_info:
                width, height, _ = segment_info[filename]
            valid_files.append((original_id, filename, width, height))
            print(f"✅ 保留: {filename} | 进度: {processed_count}/{len(tasks)}")
        else:
            print(f"❌ 排除: {filename} → {reply} | 进度: {processed_count}/{len(tasks)}")

    # 保存最终缓存
    print("\n💾 保存最终缓存...")
    save_cache_atomic()

    # 检查中断状态
    if stop_event.is_set():
        print("\n⚠️ 处理被用户中断，已保存当前进度")
        print(f"✅ 已成功处理: {processed_count} 张图")
        print(f"⏳ 未完成: {len(tasks) - processed_count} 张图")
    else:
        print(f"\n✅ 全部 {len(tasks)} 张图处理完成!")

# ======================================================
#        3. 重新编号并复制文件
# ======================================================
if valid_files:
    print("\n" + "=" * 50)
    print("🔄 重编号并复制文件...")
    image_groups = defaultdict(list)

    for original_id, filename, width, height in valid_files:
        # 确保 original_id 有效
        if original_id is None:
            continue
        image_groups[original_id].append((filename, width, height))

    total_copied = 0
    for original_id, segments in image_groups.items():
        segments.sort(key=lambda x: parse_filename(x[0])[1])
        for new_idx, (old_filename, width, height) in enumerate(segments):
            new_filename = f"sciir_img_{original_id}_{new_idx:02d}.png"
            src = os.path.join(input_dir, old_filename)
            dst = os.path.join(filtered_dir, new_filename)

            try:
                shutil.copy2(src, dst)  # 保留元数据
                total_copied += 1
                print(f"📦 复制: {old_filename} → {new_filename}")
            except Exception as e:
                print(f"❌ 复制失败 {old_filename}: {e}")

    print(f"✅ 共复制 {total_copied} 个有效文件到 {filtered_dir}")
    print("=" * 50)
else:
    print("\n⚠️ 没有有效文件需要复制")

# ======================================================
#        4. 更新 metadata（保持原逻辑）
# ======================================================
if metadata_list and valid_files:
    print("\n" + "=" * 50)
    print("📝 更新 metadata.json...")

    new_segments_map = {}

    for original_id, segments in image_groups.items():
        full_id = f"sciir_img_{original_id}"
        new_list = []
        segments.sort(key=lambda x: parse_filename(x[0])[1])
        for new_idx, (old_fname, width, height) in enumerate(segments):
            new_fname = f"sciir_img_{original_id}_{new_idx:02d}.png"
            rel_path = os.path.join("scir_dataset", "filtered_images", new_fname)
            new_list.append({
                "filename": new_fname,
                "path": rel_path,
                "width": width,
                "height": height
            })
        new_segments_map[full_id] = new_list

    updated = []
    for meta in metadata_list:
        if meta["image_id"] in new_segments_map:
            meta["segments"] = new_segments_map[meta["image_id"]]
            updated.append(meta)

    output_path = "filter_metadata.json"
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(updated, f, indent=2, ensure_ascii=False)
        print(f"✅ metadata 已更新: {output_path} | 共 {len(updated)} 条记录")
    except Exception as e:
        print(f"❌ 保存metadata失败: {e}")
    print("=" * 50)

# ======================================================
#        5. 最终保存缓存
# ======================================================
print("\n" + "=" * 50)
print("🔒 执行最终缓存保存...")
save_cache_atomic()
print("=" * 50)

print(f"\n🎉 完成！共筛选出 {len(valid_files)} 张科学图像。")
print(f"✅ 结果保存至: {filtered_dir}")
if valid_files and metadata_list:
    print(f"✅ 更新的metadata: filter_metadata.json")

# 额外诊断信息
print("\n" + "=" * 50)
print("📊 诊断信息:")
print(f"- 缓存文件: {cache_path} (大小: {os.path.getsize(cache_path) if os.path.exists(cache_path) else 0} 字节)")
print(f"- 有效图像: {len(valid_files)}")
print(f"- 总处理: {len(all_results)}")
print("=" * 50)