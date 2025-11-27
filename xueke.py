import os
import json
import shutil
import tempfile
import time

# 配置（按需修改）
metadata_path = "./scir_dataset/classified_metadata6.json"
images_dir = "./scir_dataset/filtered_images_6"
backup_path = "./scir_dataset/classified_metadata6_old.json"


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
    removed_entries = 0
    replaced_labels_count = 0  # 新增：统计替换次数

    new_metadata_list = []
    for entry in metadata_list:
        segs = entry.get("segments", [])
        # 兼容 segments 为 dict 的情况
        if isinstance(segs, dict):
            segs = [segs]
        if not isinstance(segs, list):
            segs = list(segs) if segs else []

        total_segments += len(segs)
        new_segments = []
        for seg in segs:
            labels = seg.get("labels")
            filename = seg.get("filename")
            path_in_seg = seg.get("path")

            # 1. 检查标签是否为空 (现有逻辑)
            if is_labels_empty(labels):
                # 删除对应的图像文件
                img_paths_to_try = []
                if filename:
                    img_paths_to_try.append(os.path.join(images_dir, filename))
                if path_in_seg:
                    img_paths_to_try.append(path_in_seg)
                if path_in_seg:
                    img_paths_to_try.append(os.path.join(images_dir, os.path.basename(path_in_seg)))

                deleted_any = False
                for img_path in img_paths_to_try:
                    try:
                        if img_path and os.path.exists(img_path):
                            os.remove(img_path)
                            removed_files += 1
                            deleted_any = True
                            print(f"删除图像: {img_path}")
                            break
                    except Exception as e:
                        print(f"⚠️ 删除图像失败 ({img_path}): {e}")

                removed_segments += 1
                # 不将该 segment 加入 new_segments（即删除）
            else:
                # 2. 如果标签不为空，执行替换逻辑 (新功能)
                # 检查 labels 是否为列表并包含目标字符串
                if isinstance(labels, list):
                    modified_labels = []
                    has_replaced = False
                    for lbl in labels:
                        if lbl == "ScientificConsistency":
                            modified_labels.append("ScientificLaw")
                            has_replaced = True
                            replaced_labels_count += 1
                        else:
                            modified_labels.append(lbl)

                    if has_replaced:
                        seg["labels"] = modified_labels
                        # print(f"替换标签: {entry.get('image_id')} -> {seg.get('filename')}") # 调试用

                new_segments.append(seg)

        # 如果该 entry 没有剩余 segments，则整个 entry 被删除
        if len(new_segments) == 0:
            removed_entries += 1
            print(f"已删除条目（segments 为空）：image_id={entry.get('image_id')}")
            continue

        entry["segments"] = new_segments
        new_metadata_list.append(entry)

    # 原子性保存更新后的 metadata（覆盖原文件）
    try:
        _atomic_save_json(new_metadata_list, metadata_path, retries=6, base_delay=0.3)
        print(f"✔ 已保存更新后的元数据: {metadata_path}")
    except Exception as e:
        print(f"❌ 保存更新后的元数据失败: {e}")
        print(f"⚠️ 已保留备份文件: {backup_path}")
        return

    print("----- 总结 -----")
    print(f"原始 segment 总数: {total_segments}")
    print(f"已删除 segment 数: {removed_segments}")
    print(f"已删除对应图像数: {removed_files}")
    print(f"已删除条目（segments 为空）数: {removed_entries}")
    print(f"已替换标签 (Consistency->Law) 次数: {replaced_labels_count}")
    print("操作完成。")


if __name__ == "__main__":
    main()