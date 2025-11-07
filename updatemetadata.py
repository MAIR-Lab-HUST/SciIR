import os
import json
from collections import defaultdict
from PIL import Image  # ✅ 用于读取图片尺寸

# ===== 路径设置 =====
filtered_dir = "./scir_dataset/filtered_images"
metadata_path = "./scir_dataset/metadata.json"
updated_metadata_path = "./scir_dataset/metadata_updated.json"

# ===== 加载元数据 =====
print("📂 正在加载 metadata.json ...")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"未找到元数据文件: {metadata_path}")

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata_list = json.load(f)

# ===== 扫描实际存在的图片 =====
print("🔍 扫描实际存在的图片...")
existing_files = set(
    f for f in os.listdir(filtered_dir)
    if f.startswith("sciir_img_") and f.endswith(".png")
)
print(f"找到 {len(existing_files)} 张有效图片")

# ===== 解析文件名，映射 image_id -> 文件 =====
def parse_image_id(filename):
    # sciir_img_000123_01.png -> sciir_img_000123
    parts = filename.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return None

image_groups = defaultdict(list)
for fname in existing_files:
    image_id = parse_image_id(fname)
    if image_id:
        image_groups[image_id].append(fname)

# ===== 辅助函数：安全获取图片尺寸 =====
def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None

# ===== 更新 metadata =====
print("🧩 开始更新 metadata ...")
updated_metadata_list = []
updated_count = 0
deleted_count = 0

for metadata in metadata_list:
    image_id = metadata["image_id"]
    if image_id in image_groups:
        new_segments = []

        # 对每个现有 segment 重新生成记录
        for fname in sorted(image_groups[image_id]):

            img_path = os.path.join(filtered_dir, fname)
            width, height = get_image_size(img_path)

            relative_path = os.path.join("scir_dataset", "filtered_images", fname)
            new_segments.append({
                "filename": fname,
                "path": relative_path,
                "width": width,
                "height": height
            })

        metadata["segments"] = new_segments
        updated_metadata_list.append(metadata)
        updated_count += 1
    else:
        deleted_count += 1
        print(f"🗑️ 删除: {image_id} (对应图片已全部移除)")

# ===== 保存新的 metadata =====
with open(updated_metadata_path, "w", encoding="utf-8") as f:
    json.dump(updated_metadata_list, f, indent=2, ensure_ascii=False)

print(f"\n✅ 更新完成!")
print(f"保留 {updated_count} 条记录，删除 {deleted_count} 条记录。")
print(f"新文件已保存至: {updated_metadata_path}")
