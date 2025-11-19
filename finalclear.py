import os
import json
import shutil
import tempfile
import time

# 配置（按需修改）
metadata_path = "D:\\code\\sci\\ok\\classified_metadata_1b.json"
images_dir = "D:\\code\\sci\\ok\\filtered_images_1b"
backup_path = "D:\\code\\sci\\ok\\classified_metadata_1b_final.json"

def _atomic_save_json(obj, path, retries=5, base_delay=0.3):
    dirn = os.path.dirname(path) or "."
    for attempt in range(retries):
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path), dir=dirn, text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                json.dump(obj, tf, ensure_ascii=False, indent=2)
                tf.flush()
                os.fsync(tf.fileno())
            os.replace(tmp_path, path)
            return
        except PermissionError:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise
        except Exception:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

def is_labels_empty(labels):
    # 视为空的情况：None / 空字符串 / 空列表
    if labels is None:
        return True
    if isinstance(labels, str) and not labels.strip():
        return True
    if isinstance(labels, (list, tuple)) and len(labels) == 0:
        return True
    return False

def main():
    if not os.path.exists(metadata_path):
        print(f"❌ 找不到元数据文件: {metadata_path}")
        return

    # 备份原始文件
    try:
        shutil.copy2(metadata_path, backup_path)
        print(f"✔ 已备份原始元数据到: {backup_path}")
    except Exception as e:
        print(f"⚠️ 无法备份元数据: {e}")

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取元数据: {e}")
        return

    total_segments = 0
    removed_segments = 0
    removed_files = 0

    for entry in metadata_list:
        segments = entry.get("segments", [])
        total_segments += len(segments)
        new_segments = []
        for seg in segments:
            labels = seg.get("labels")
            filename = seg.get("filename")
            if is_labels_empty(labels):
                # 删除对应的图像文件（如果存在）
                if filename:
                    img_path = os.path.join(images_dir, filename)
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            removed_files += 1
                            print(f"删除图像: {img_path}")
                        else:
                            print(f"图像不存在，跳过删除: {img_path}")
                    except Exception as e:
                        print(f"⚠️ 删除图像失败 ({img_path}): {e}")
                removed_segments += 1
                # 不将该 segment 加入 new_segments（即删除）
            else:
                new_segments.append(seg)
        entry["segments"] = new_segments

    # 原子性保存更新后的 metadata
    try:
        _atomic_save_json(metadata_list, metadata_path, retries=6, base_delay=0.3)
        print(f"✔ 已保存更新后的元数据: {metadata_path}")
    except Exception as e:
        print(f"❌ 保存更新后的元数据失败: {e}")
        print(f"⚠️ 已保留备份文件: {backup_path}")
        return

    print("----- 总结 -----")
    print(f"原始 segment 总数: {total_segments}")
    print(f"已删除 segment 数: {removed_segments}")
    print(f"已删除对应图像数: {removed_files}")
    print("操作完成。")

if __name__ == "__main__":
    main()