import json
import os
from collections import Counter

# ================= 配置区域 =================
# 你的目标 JSON 文件路径
JSON_FILE_PATH = "./scir_dataset/classified_abstracts.json"


# ===========================================

def main():
    print("=" * 60)
    print(f"📊 正在分析文件: {JSON_FILE_PATH}")
    print("=" * 60)

    # 1. 检查文件是否存在
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ 错误: 找不到文件 {JSON_FILE_PATH}")
        return

    # 2. 读取 JSON
    try:
        with open(JSON_FILE_PATH, "r", encoding='utf-8') as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return

    total_count = len(data_list)
    print(f"📚 总数据量: {total_count} 条")

    # 3. 提取分类并统计
    # 注意：如果某条数据没有 'subject_category' 字段，或者为 None，归类为 "Uncategorized"
    categories = []
    for item in data_list:
        cat = item.get("subject_category")
        if cat is None:
            categories.append("Uncategorized (None)")
        else:
            categories.append(cat)

    # 使用 Counter 进行计数
    counter = Counter(categories)

    # 按数量从多到少排序
    sorted_stats = counter.most_common()

    # 4. 打印详细报告
    print("\n📈 学科分布统计报告:")
    print("-" * 60)
    print(f"{'Category Name':<40} | {'Count':<8} | {'Percent':<8} | {'Chart'}")
    print("-" * 60)

    for category, count in sorted_stats:
        percentage = (count / total_count) * 100

        # 简单的 ASCII 柱状图 (每 2% 一个 #)
        bar_length = int(percentage / 2)
        bar_chart = "#" * bar_length

        print(f"{category:<40} | {count:<8} | {percentage:6.2f}% | {bar_chart}")

    print("-" * 60)

    # 5. 简单的完整性检查
    uncategorized_count = counter.get("Uncategorized (None)", 0)
    if uncategorized_count > 0:
        print(
            f"\n⚠️以此注意: 有 {uncategorized_count} 条数据 ({uncategorized_count / total_count * 100:.1f}%) 未成功分类。")
        print("建议: 检查这些数据的 'article_abstract' 是否为空，或运行分类脚本补全。")
    else:
        print("\n✅ 所有数据均已成功分类。")


if __name__ == "__main__":
    main()