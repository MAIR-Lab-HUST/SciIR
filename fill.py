from PIL import Image
import os
import concurrent.futures
from collections import Counter
import json
from pathlib import Path

def get_fill_color_for_padding(img, target_size=1024):
    """
    根据图像宽高比，只从需要填充的方向（上下 或 左右）取边缘像素：
    - 如果边缘中 >70% 像素是同一颜色，则直接用该颜色；
    - 否则使用平均颜色。
    返回 RGB 填充颜色。
    """
    w, h = img.size

    if w == 0 or h == 0:
        return (255, 255, 255)

    # 转为 RGB 模式用于颜色计算（忽略透明度）
    if img.mode in ('RGBA', 'LA', 'P'):
        rgb_img = img.convert('RGB')
    else:
        rgb_img = img.convert('RGB')

    edge_width = 1

    # 计算缩放比例，判断哪个方向需要填充
    scale_w = target_size / w
    scale_h = target_size / h
    ratio = min(scale_w, scale_h)

    new_w = int(w * ratio)
    new_h = int(h * ratio)

    pixels = []

    if new_w == target_size and new_h < target_size:
        # 高度不足 → 上下填充 → 取上下边缘
        top_edge = rgb_img.crop((0, 0, w, edge_width))
        bottom_edge = rgb_img.crop((0, h - edge_width, w, h))
        pixels.extend(list(top_edge.getdata()))
        pixels.extend(list(bottom_edge.getdata()))

    elif new_h == target_size and new_w < target_size:
        # 宽度不足 → 左右填充 → 取左右边缘
        left_edge = rgb_img.crop((0, 0, edge_width, h))
        right_edge = rgb_img.crop((w - edge_width, 0, w, h))
        pixels.extend(list(left_edge.getdata()))
        pixels.extend(list(right_edge.getdata()))

    else:
        # fallback：四周边缘
        top = rgb_img.crop((0, 0, w, edge_width))
        bottom = rgb_img.crop((0, h - edge_width, w, h))
        left = rgb_img.crop((0, 0, edge_width, h))
        right = rgb_img.crop((w - edge_width, 0, w, h))
        for region in [top, bottom, left, right]:
            pixels.extend(list(region.getdata()))

    if not pixels:
        return (255, 255, 255)  # fallback 白色

    total = len(pixels)
    if total == 0:
        return (255, 255, 255)

    # 统计最常见颜色
    counter = Counter(pixels)
    most_common_color, count = counter.most_common(1)[0]
    if count / total >= 0.55:
        return most_common_color

    # 否则使用平均颜色
    avg_r = int(sum(p[0] for p in pixels) / total)
    avg_g = int(sum(p[1] for p in pixels) / total)
    avg_b = int(sum(p[2] for p in pixels) / total)
    return (avg_r, avg_g, avg_b)


def resize_and_pad_adaptive(img_path, output_path, target_size=1024):
    """
    等比缩放 + 按需填充，填充色来自对应方向的边缘（智能选择：众数或平均）。
    输出为 RGB PNG（丢弃透明通道）。
    """
    try:
        with Image.open(img_path) as img:
            fill_color = get_fill_color_for_padding(img, target_size)

            ratio = min(target_size / img.width, target_size / img.height)
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)

            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            new_img = Image.new('RGB', (target_size, target_size), fill_color)

            x = (target_size - new_width) // 2
            y = (target_size - new_height) // 2

            if img.mode in ('RGBA', 'LA'):
                # 使用 alpha 通道作为 mask
                alpha = img_resized.split()[-1] if img_resized.mode in ('RGBA', 'LA') else None
                new_img.paste(img_resized, (x, y), mask=alpha)
            else:
                new_img.paste(img_resized, (x, y))

            new_img.save(output_path, format='PNG')
    except Exception as e:
        raise e  # 由调用者捕获


def process_single_file(filename, input_folder, output_folder, target_size=1024):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
    try:
        resize_and_pad_adaptive(input_path, output_path, target_size)
        print(f"✅ Processed: {filename}")
    except Exception as e:
        print(f"❌ Error on {filename}: {e}")


# ===== 批量多线程处理 =====
if __name__ == "__main__":
    input_folder = "./scir_dataset/cropped_images"
    output_folder = "./scir_dataset/filled_images"
    target_size = 1024
    max_workers = os.cpu_count()

    os.makedirs(output_folder, exist_ok=True)

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file, f, input_folder, output_folder, target_size)
            for f in files
        ]
        concurrent.futures.wait(futures)

    print("🎉 All images processed!")
    
    # === 更新 metadata 中的 segments 尺寸为 1024x1024 ===
    print("\n📝 Updating segments dimensions in metadata...")
    metadata_path = Path(__file__).parent / "scir_dataset/cropped_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 更新所有 segments 的 width 和 height
        total_updated = 0
        for item in metadata:
            if "segments" in item and item["segments"]:
                for segment in item["segments"]:
                    segment["width"] = target_size
                    segment["height"] = target_size
                    total_updated += 1
        
        # 保存更新后的 metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Updated {total_updated} segments to {target_size}x{target_size}")
    else:
        print(f"⚠️  Metadata file not found: {metadata_path}")
    
    print("🎉 All done!")