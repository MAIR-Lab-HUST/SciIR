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

# ✅ 初始化客户端
client = OpenAI(
    api_key="sk-dfiqrcjtmmnjntnapwfgdtkvcljspkzowxfnqzvfoqqbuljf",
    base_url="https://api.siliconflow.cn/v1",
)

# ✅ 配置：筛选范围控制
# 第一次运行设置为 "less_than_10000"，第二次运行设置为 "greater_or_equal_10000"
FILTER_MODE = "greater_or_equal_10000"  # 可选值: "less_than_10000", "greater_or_equal_10000", "all"
ID_THRESHOLD = 10000  # original_id 的阈值

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

input_dir = "./scir_dataset/filled_images_6"

filtered_dir = "scir_dataset/filtered_images_6"
metadata_path = "./scir_dataset/cropped_metadata_6.json"
os.makedirs(filtered_dir, exist_ok=True)

# ✅ 加载缓存
cache_path = "filter_cache.json"
cache_lock = threading.Lock()
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding='utf-8') as f:
        cache = json.load(f)
else:
    cache = {}


def parse_filename(filename):
    """解析文件名，提取原始图片号和子图号"""
    match = re.match(r"sciir_img_(\d{6})_(\d{2})\.png", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def should_process_file(filename):
    """根据配置的筛选模式判断是否应该处理该文件"""
    original_id, _ = parse_filename(filename)
    if not original_id:
        return False

    original_id_int = int(original_id)

    if FILTER_MODE == "less_than_10000":
        return original_id_int < ID_THRESHOLD
    elif FILTER_MODE == "greater_or_equal_10000":
        return original_id_int >= ID_THRESHOLD
    elif FILTER_MODE == "all":
        return True
    else:
        print(f"⚠️ 未知的筛选模式: {FILTER_MODE}，默认处理所有文件")
        return True


def process_image(filename):
    """单张图片识别逻辑，添加重试机制处理速率限制"""
    img_path = os.path.join(input_dir, filename)

    # 使用完整文件名作为缓存key
    if filename in cache:
        return filename, cache[filename]

    with open(img_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}},
            {"type": "text", "text": "这张图像是否为'科学图像'？"}
        ]}
    ]

    max_retries = 5  # 最大重试次数
    base_delay = 4  # 基础延迟时间(秒)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct",
                messages=messages,
                temperature=0
            )
            reply = response.choices[0].message.content.strip()

            # 并发安全地更新并立即保存 cache
            with cache_lock:
                cache[filename] = reply
                tmp_path = cache_path + ".tmp"
                with open(tmp_path, "w", encoding='utf-8') as cf:
                    json.dump(cache, cf, ensure_ascii=False, indent=2)
                os.replace(tmp_path, cache_path)

            return filename, reply

        except Exception as e:
            error_msg = str(e)
            # 专门处理速率限制错误
            if "429" in error_msg or "RPM limit reached" in error_msg:
                if attempt < max_retries - 1:
                    # 指数退避: 2, 4, 8, 16秒...
                    wait_time = base_delay * (8 ** attempt)
                    print(
                        f"⚠️ 速率限制触发: {filename} (尝试 {attempt + 1}/{max_retries}) - 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    final_error = f"❌ 达到最大重试次数后仍失败: {e}"
                    print(final_error)
                    return filename, final_error
            else:
                # 其他错误直接返回
                return filename, f"❌ Error: {e}"

    # 理论上不会执行到这里
    return filename, "❌ 未知错误"


# ✅ 1. 首先加载metadata.json并创建segment信息映射
print("加载元数据文件...")
segment_info = {}  # 映射: segment filename -> (width, height)
metadata_list = []

if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding='utf-8') as f:
        metadata_list = json.load(f)

    # 构建segment信息映射
    for metadata in metadata_list:
        image_id = metadata["image_id"]
        for segment in metadata["segments"]:
            segment_info[segment["filename"]] = (
                segment["width"],
                segment["height"],
                image_id
            )
    print(f"成功加载 {len(metadata_list)} 条元数据，包含 {len(segment_info)} 个segment")
else:
    print("元数据文件不存在，将跳过metadata更新步骤")

# ✅ 2. 并行执行图像筛选
all_results = []  # 存储所有结果 [(filename, reply), ...]
valid_files = []  # 存储有效文件 [(original_id, filename, width, height), ...]

# 打印当前筛选模式
print(f"\n📋 当前筛选模式: {FILTER_MODE}")
if FILTER_MODE == "less_than_10000":
    print(f"   只处理 original_id < {ID_THRESHOLD} 的图片")
elif FILTER_MODE == "greater_or_equal_10000":
    print(f"   只处理 original_id >= {ID_THRESHOLD} 的图片")
else:
    print("   处理所有图片")

print("\n🔍 开始处理图像...")
with ThreadPoolExecutor(max_workers=600) as executor:
    # 提交所有任务
    tasks = {}
    skipped_count = 0
    for fname in os.listdir(input_dir):
        if fname.startswith("sciir") and fname.endswith(".png"):
            # ✅ 根据配置的筛选模式判断是否处理该文件
            if should_process_file(fname):
                tasks[executor.submit(process_image, fname)] = fname
            else:
                skipped_count += 1

    print(f"📊 共 {len(tasks)} 个文件将被处理，{skipped_count} 个文件被跳过（不符合筛选条件）")

    # 处理结果
    for future in as_completed(tasks):
        filename, reply = future.result()
        all_results.append((filename, reply))

        # 严格匹配"科学图像"
        if reply.strip() == "科学图像":
            # 解析原始图片号
            original_id, _ = parse_filename(filename)
            if original_id:
                # 获取segment的width和height（如果metadata存在）
                width, height = None, None
                if filename in segment_info:
                    width, height, _ = segment_info[filename]

                valid_files.append((original_id, filename, width, height))
                print(f"✅ 保留: {filename} (原始ID: {original_id})")
            else:
                print(f"⚠️ 跳过无效命名文件: {filename}")
        else:
            print(f"❌ 排除: {filename} → {reply}")

# ✅ 3. 按原始图片号分组并重新编号
print("\n🔄 开始重新编号并复制文件...")
image_groups = defaultdict(list)

# 按原始图片号分组
for original_id, filename, width, height in valid_files:
    image_groups[original_id].append((filename, width, height))

# 处理每个组
for original_id, segments in image_groups.items():
    # 按原始子图号排序（确保顺序一致）
    segments.sort(key=lambda x: parse_filename(x[0])[1])

    # 重新编号并复制
    for new_idx, (old_filename, width, height) in enumerate(segments):
        new_filename = f"sciir_img_{original_id}_{new_idx:02d}.png"
        src_path = os.path.join(input_dir, old_filename)
        dst_path = os.path.join(filtered_dir, new_filename)

        shutil.copy(src_path, dst_path)
        print(f"复制: {old_filename} → {new_filename} (组: {original_id})")

# ✅ 4. 更新metadata.json
if os.path.exists(metadata_path) and metadata_list:
    print("\n开始更新metadata.json...")

    # 创建一个字典来跟踪每个image_id的新segments
    new_segments_map = {}

    # 为每个保留的segment构建新结构（使用完整的 image_id 格式）
    for original_id, segments in image_groups.items():
        full_image_id = f"sciir_img_{original_id}"  # ✅ 转换为完整格式
        new_segments = []
        segments.sort(key=lambda x: parse_filename(x[0])[1])  # 排序

        for new_idx, (old_filename, width, height) in enumerate(segments):
            new_filename = f"sciir_img_{original_id}_{new_idx:02d}.png"
            # ✅ 添加path字段，存储相对路径
            relative_path = os.path.join("scir_dataset", "filtered_images", new_filename)

            new_segments.append({
                "filename": new_filename,
                "path": relative_path,  # ✅ 新增path字段
                "width": width,
                "height": height
            })
        new_segments_map[full_image_id] = new_segments

    # ✅ 过滤并更新元数据（删除没有保留图片的记录）
    updated_metadata_list = []
    updated_count = 0
    deleted_count = 0

    for metadata in metadata_list:
        image_id = metadata["image_id"]
        if image_id in new_segments_map:
            # 保留并更新 segments
            metadata["segments"] = new_segments_map[image_id]
            updated_metadata_list.append(metadata)
            updated_count += 1
        else:
            # 删除这条记录
            deleted_count += 1
            print(f"删除: {image_id} (该图像的所有子图都被筛除)")

    # ✅ 确保输出目录存在
    output_metadata_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    final_metadata_path = os.path.join(output_metadata_dir, "filtered_metadata.json")

    try:
        with open(final_metadata_path, "w", encoding="utf-8") as f:
            json.dump(updated_metadata_list, f, indent=2, ensure_ascii=False)
        print(f"✅ 成功更新元数据文件，共保留 {updated_count} 条记录")
        print(f"📌 已保存至: {final_metadata_path}")
    except Exception as e:
        print(f"❌ 写入元数据文件失败: {e}")

# ✅ 5. 保存缓存
with open(cache_path, "w", encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

print(f"\n🎉 处理完成! 共筛选出 {len(valid_files)} 张科学图像，保存至 {filtered_dir}")

