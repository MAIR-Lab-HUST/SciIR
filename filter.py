import os
import base64
import shutil
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from collections import defaultdict

# ✅ 初始化客户端
client = OpenAI(
    api_key="sk-eGILuyIosedfyzKZB2IMycqSAaywcJpgbzpi3EGxuzmj53mM",
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

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

input_dir = "./scir_dataset/filtered_images"
filtered_dir = "./scir_dataset/filtered2_images"
metadata_path = "./scir_dataset/metadata.json"
os.makedirs(filtered_dir, exist_ok=True)

# ✅ 加载缓存
cache_path = "filter_cache.json"
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


def process_image(filename):
    """单张图片识别逻辑"""
    img_path = os.path.join(input_dir, filename)

    # 使用完整文件名作为缓存key
    if filename in cache:
        return filename, cache[filename]

    with open(img_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图像是否为'科学图像'？"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="internvl3.5-latest",
            messages=messages,
            temperature=0
        )
        reply = response.choices[0].message.content.strip()
        cache[filename] = reply
        return filename, reply

    except Exception as e:
        return filename, f"❌ Error: {e}"


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

print("\n🔍 开始处理图像...")
with ThreadPoolExecutor(max_workers=50) as executor:
    # 提交所有任务
    tasks = {}
    for fname in os.listdir(input_dir):
        if fname.startswith("sciir") and fname.endswith(".png"):
            tasks[executor.submit(process_image, fname)] = fname

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

    # 保存更新后的metadata
    updated_metadata_path = "./scir_dataset/metadata_updated.json"
    with open(updated_metadata_path, "w", encoding='utf-8') as f:
        json.dump(updated_metadata_list, f, indent=2, ensure_ascii=False)

    print(f"成功更新 {updated_count} 条元数据记录")
    print(f"删除 {deleted_count} 条元数据记录")
    print(f"更新后的元数据已保存至: {updated_metadata_path}")
else:
    print("\n元数据文件不存在或为空，跳过metadata更新步骤")

# ✅ 5. 保存缓存
with open(cache_path, "w", encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

print(f"\n🎉 处理完成! 共筛选出 {len(valid_files)} 张科学图像，保存至 {filtered_dir}")
