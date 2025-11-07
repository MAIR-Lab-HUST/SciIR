from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import json
import warnings

# 忽略PIL/libpng的ICC profile警告
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
import logging
logging.getLogger('PIL').setLevel(logging.ERROR)

# === 路径设置 ===
DOWNLOAD_PATH = Path(__file__).parent / "models"
SAMPLES_ROOT = Path(__file__).parent / "scir_dataset/images"
CROPS_DIR = Path(__file__).parent / "scir_dataset/cropped_images"
METADATA_PATH = Path(__file__).parent / "scir_dataset/metadata.json"

CROPS_DIR.mkdir(exist_ok=True)

# === 读取metadata ===
print("Loading metadata.json...")
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 创建image_id到metadata索引的映射
metadata_map = {item['image_id']: idx for idx, item in enumerate(metadata)}

# === 下载模型 ===
model_path = DOWNLOAD_PATH / "yolo11n_doc_layout.pt"

# === 初始化模型（GPU优化）===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO(model_path)
model.to(device)

# 预热模型（可选，首次推理会慢）
if device == 'cuda':
    dummy_img = np.zeros((960, 960, 3), dtype=np.uint8)
    _ = model(dummy_img, verbose=False)

# === 读取图片 ===
images = sorted(SAMPLES_ROOT.glob("*.png"))

# 用于存储每张图片的分割信息
segments_info = {}


# === 图片验证函数 ===
def validate_image(img_path):
    """验证图片是否可读，修复ICC profile问题"""
    try:
        # 尝试用cv2读取
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        
        # 检查图片尺寸是否有效
        if img.shape[0] == 0 or img.shape[1] == 0:
            return False
        
        return True
    except Exception as e:
        print(f"Warning: Image validation failed for {img_path}: {e}")
        return False


# === 批处理 + 多线程保存 ===
def process_batch(batch_paths, executor):
    """处理一批图片"""
    # 预验证所有图片
    valid_paths = [p for p in batch_paths if validate_image(p)]
    
    if not valid_paths:
        print(f"Warning: No valid images in batch")
        return []
    
    if len(valid_paths) < len(batch_paths):
        invalid_paths = set(batch_paths) - set(valid_paths)
        for p in invalid_paths:
            print(f"Skipping corrupted image: {p}")
    
    try:
        # 批量推理
        batch_results = model(
            [str(p) for p in valid_paths],
            conf=0.15,
            iou=0.6,
            verbose=False,
            device=device,
            half=(device == 'cuda'),  # GPU使用FP16
            batch=len(valid_paths),
            imgsz=960
        )
    except Exception as e:
        print(f"Batch processing failed: {e}")
        print("Falling back to individual processing...")
        # 如果批处理失败，逐个处理
        save_tasks = []
        for img_path in valid_paths:
            try:
                tasks = process_single_image(img_path, executor)
                save_tasks.extend(tasks)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
        return save_tasks

    save_tasks = []

    for img_path, results in zip(valid_paths, batch_results):
        if len(results.boxes) == 0:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # 从文件名提取image_id (例如: sciir_img_000000.png -> sciir_img_000000)
        image_id = img_path.stem

        # 初始化该图片的segments列表
        if image_id not in segments_info:
            segments_info[image_id] = []

        pic_count = 0  # 用于给 picture 编号

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]

            # ✅ 只处理类别为 "picture" 的检测框
            if cls_name != "Picture":
                continue

            pic_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 边界检查（防止坐标越界）
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue  # 跳过无效框

            # === 计算特征参数 ===
            crop_w = x2 - x1
            crop_h = y2 - y1
            box_area = crop_w * crop_h
            img_area = w * h
            area_ratio = box_area / img_area
            aspect_ratio = crop_w / crop_h if crop_h > 0 else 0

            # === 筛选逻辑 ===
            # 1️⃣ 跳过面积大于75%但小于整图的框
            if 0.75 < area_ratio < 0.9:
                continue

            # 2️⃣ 跳过宽高比不在 [0.33, 3] 范围内的
            if not (0.33 <= aspect_ratio <= 3):
                continue

            # 3️⃣ 跳过太小的裁剪区域
            if crop_w < 128 or crop_h < 128:
                continue

            crop = image[y1:y2, x1:x2].copy()

            # 新的命名格式: sciir_img_xxxxxx_01.png
            crop_filename = f"{image_id}_{pic_count:02d}.png"
            crop_path = CROPS_DIR / crop_filename

            # 记录分割信息
            segments_info[image_id].append({
                "filename": crop_filename,
                "width": crop_w,
                "height": crop_h
            })

            future = executor.submit(cv2.imwrite, str(crop_path), crop)
            save_tasks.append((crop_path, future))

    return save_tasks


def process_single_image(img_path, executor):
    """处理单张图片（用于批处理失败时的fallback）"""
    save_tasks = []
    
    try:
        # 单张图片推理
        results = model(
            str(img_path),
            conf=0.15,
            iou=0.6,
            verbose=False,
            device=device,
            half=(device == 'cuda'),
            imgsz=960
        )[0]  # 返回的是列表，取第一个结果
        
        if len(results.boxes) == 0:
            return save_tasks
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            return save_tasks
        
        # 从文件名提取image_id
        image_id = img_path.stem
        
        # 初始化该图片的segments列表
        if image_id not in segments_info:
            segments_info[image_id] = []
        
        pic_count = 0
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            
            if cls_name != "Picture":
                continue
            
            pic_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 边界检查
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 计算特征参数
            crop_w = x2 - x1
            crop_h = y2 - y1
            box_area = crop_w * crop_h
            img_area = w * h
            area_ratio = box_area / img_area
            aspect_ratio = crop_w / crop_h if crop_h > 0 else 0
            
            # 筛选逻辑
            if 0.75 < area_ratio < 0.9:
                continue
            
            if not (0.33 <= aspect_ratio <= 3):
                continue
            
            if crop_w < 128 or crop_h < 128:
                continue
            
            crop = image[y1:y2, x1:x2].copy()
            crop_filename = f"{image_id}_{pic_count:02d}.png"
            crop_path = CROPS_DIR / crop_filename
            
            segments_info[image_id].append({
                "filename": crop_filename,
                "width": crop_w,
                "height": crop_h
            })
            
            future = executor.submit(cv2.imwrite, str(crop_path), crop)
            save_tasks.append((crop_path, future))
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    
    return save_tasks


# === 主处理循环 ===
batch_size = 16 if device == 'cuda' else 2  # GPU可以处理更大批次

with ThreadPoolExecutor(max_workers=6) as executor:
    all_save_tasks = []

    # 使用进度条
    for i in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
        batch = images[i:i + batch_size]
        tasks = process_batch(batch, executor)
        all_save_tasks.extend(tasks)

    # 等待所有保存完成
    print("\nWaiting for saves to complete...")
    for crop_path, future in tqdm(all_save_tasks, desc="Saving crops"):
        success = future.result()
        if not success:
            print(f"Failed to save: {crop_path}")

# === 更新metadata.json ===
print("\nUpdating cropped_metadata.json...")

CROPPED_METADATA_PATH = METADATA_PATH.parent / "cropped_metadata.json"

# 创建新的metadata列表，复制所有原字段并添加segments
cropped_metadata = []
for item in metadata:
    # 深拷贝原metadata的所有字段
    new_item = item.copy()
    
    # 如果该图片有分割信息，添加segments字段
    image_id = item.get('image_id', '')
    if image_id in segments_info:
        new_item["segments"] = segments_info[image_id]
    else:
        # 没有分割信息的图片，添加空的segments列表
        new_item["segments"] = []
    
    cropped_metadata.append(new_item)

# 保存更新后的metadata为 cropped_metadata.json
with open(CROPPED_METADATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(cropped_metadata, f, ensure_ascii=False, indent=2)

print(f"\n✅ Processing complete!")
print(f"📁 Crops saved to: {CROPS_DIR}")
print(f"📝 Metadata saved to: {CROPPED_METADATA_PATH}")
print(f"🖼️  Total images with segments: {len(segments_info)}")
total_segments = sum(len(segs) for segs in segments_info.values())
print(f"✂️  Total segments created: {total_segments}")