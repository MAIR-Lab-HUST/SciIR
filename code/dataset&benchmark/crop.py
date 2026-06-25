import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import json
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.getLogger('PIL').setLevel(logging.ERROR)

warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

# === Paths ===
DOWNLOAD_PATH = Path(__file__).parent / "models"
SAMPLES_ROOT = Path(__file__).parent / "scir_dataset/images"
CROPS_DIR = Path(__file__).parent / "scir_dataset/cropped_images"
METADATA_PATH = Path(__file__).parent / "scir_dataset/metadata.json"

CROPS_DIR.mkdir(exist_ok=True, parents=True)


def safe_imread(img_path):

    try:

        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert('RGB')

            img_array = np.array(pil_img)

            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:

        return None



print("Loading metadata.json...")
try:
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_PATH}")
    sys.exit(1)


model_path = DOWNLOAD_PATH / "yolo11n_doc_layout.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    model = YOLO(model_path)
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure model exists at: {model_path}")
    sys.exit(1)


if device == 'cuda':
    dummy_img = np.zeros((960, 960, 3), dtype=np.uint8)
    _ = model(dummy_img, verbose=False)


images = sorted(SAMPLES_ROOT.glob("*.png"))
segments_info = {}



def validate_image(img_path):

    img = safe_imread(img_path)
    if img is None:
        return False
    if img.shape[0] == 0 or img.shape[1] == 0:
        return False
    return True



def process_single_image(img_path, executor):
    save_tasks = []
    try:

        results = model(
            str(img_path),
            conf=0.15,
            iou=0.6,
            verbose=False,
            device=device,
            half=(device == 'cuda'),
            imgsz=960
        )[0]

        if len(results.boxes) == 0:
            return save_tasks


        image = safe_imread(img_path)
        if image is None:
            return save_tasks

        image_id = img_path.stem
        if image_id not in segments_info:
            segments_info[image_id] = []

        pic_count = 0
        h, w = image.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if results.names[cls_id] != "Picture":
                continue

            pic_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])


            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue


            crop_w = x2 - x1
            crop_h = y2 - y1
            area_ratio = (crop_w * crop_h) / (w * h)
            aspect_ratio = crop_w / crop_h if crop_h > 0 else 0


            if 0.75 < area_ratio < 0.9: continue
            if not (0.33 <= aspect_ratio <= 3): continue
            if crop_w < 128 or crop_h < 128: continue


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
        print(f"Error processing single {img_path}: {e}")

    return save_tasks



def process_batch(batch_paths, executor):

    valid_paths = [p for p in batch_paths if validate_image(p)]

    if not valid_paths:
        return []

    try:

        batch_results = model(
            [str(p) for p in valid_paths],
            conf=0.15,
            iou=0.6,
            verbose=False,
            device=device,
            half=(device == 'cuda'),
            batch=len(valid_paths),
            imgsz=960
        )
    except Exception as e:
        print(f"Batch inference failed, falling back to single: {e}")
        tasks = []
        for p in valid_paths:
            tasks.extend(process_single_image(p, executor))
        return tasks

    save_tasks = []

    for img_path, results in zip(valid_paths, batch_results):
        if len(results.boxes) == 0:
            continue


        image = safe_imread(img_path)
        if image is None:
            continue

        image_id = img_path.stem
        if image_id not in segments_info:
            segments_info[image_id] = []

        pic_count = 0
        h, w = image.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if results.names[cls_id] != "Picture":
                continue

            pic_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop_w = x2 - x1
            crop_h = y2 - y1
            area_ratio = (crop_w * crop_h) / (w * h)
            aspect_ratio = crop_w / crop_h if crop_h > 0 else 0

            if 0.75 < area_ratio < 0.9: continue
            if not (0.33 <= aspect_ratio <= 3): continue
            if crop_w < 128 or crop_h < 128: continue

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

    return save_tasks



if __name__ == "__main__":
    print(f"Found {len(images)} images.")

    batch_size = 16 if device == 'cuda' else 4

    with ThreadPoolExecutor(max_workers=6) as executor:
        all_save_tasks = []

        for i in tqdm(range(0, len(images), batch_size), desc="Processing"):
            batch = images[i:i + batch_size]
            tasks = process_batch(batch, executor)
            all_save_tasks.extend(tasks)

        print("\nWaiting for I/O operations to complete...")

        for crop_path, future in tqdm(all_save_tasks, desc="Saving"):
            try:
                future.result()
            except Exception as e:
                print(f"Failed to save {crop_path}: {e}")


    print("\nUpdating cropped_metadata.json...")
    CROPPED_METADATA_PATH = METADATA_PATH.parent / "cropped_metadata.json"

    cropped_metadata = []
    for item in metadata:
        new_item = item.copy()
        image_id = item.get('image_id', '')
        new_item["segments"] = segments_info.get(image_id, [])
        cropped_metadata.append(new_item)

    with open(CROPPED_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(cropped_metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done!")
    print(f"📁 Crops: {CROPS_DIR}")
    print(f"📝 Metadata: {CROPPED_METADATA_PATH}")
    print(f"✂️  Total segments: {sum(len(v) for v in segments_info.values())}")