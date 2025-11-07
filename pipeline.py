"""
科学图像数据集自动化采集流水线
任务：
1. 自动从Nature抓取论文图像
2. 增加学科、图像尺寸标签
"""
import os
import sys
import json
import logging
import time
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collection.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


def check_requirements():
    """检查依赖包"""
    required_packages = {
        'requests': 'requests',
        'bs4': 'beautifulsoup4',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pylatexenc': 'pylatexenc'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            logging.info(f"✓ {package}")
        except ImportError:
            missing.append(package)
            logging.warning(f"✗ {package} 未安装")
    
    if missing:
        logging.error(f"\n缺少依赖包: {', '.join(missing)}")
        logging.error(f"请运行: pip install {' '.join(missing)}")
        return False
    
    logging.info("所有依赖包已安装\n")
    return True


def run_collection():
    """运行数据采集"""
    logging.info("="*60)
    logging.info("开始科学图像数据集自动化采集")
    logging.info("="*60)
    logging.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 导入主程序
    try:
        from test import main as collect_main
        from test import TARGET_IMAGE_COUNT
        
        logging.info(f"目标图片数量: {TARGET_IMAGE_COUNT}")
        logging.info("采集功能:")
        logging.info("  ✓ 自动从Nature抓取论文图像")
        logging.info("  ✓ 学科标签提取")
        
        # 执行采集
        start_time = time.time()
        collect_main()
        elapsed_time = time.time() - start_time
        
        logging.info("\n" + "="*60)
        logging.info("采集完成！")
        logging.info("="*60)
        logging.info(f"总耗时: {elapsed_time/60:.2f} 分钟")
        
        return True
        
    except Exception as e:
        logging.error(f"采集过程出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def verify_results():
    """验证采集结果"""
    logging.info("\n开始验证采集结果...")
    
    output_dir = "scir_dataset"
    images_dir = os.path.join(output_dir, "images")
    metadata_file = os.path.join(output_dir, "metadata.json")
    progress_file = os.path.join(output_dir, "progress.json")
    
    results = {
        "images_collected": 0,
    }
    
    # 检查图像目录
    if os.path.exists(images_dir):
        results["images_collected"] = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
        logging.info(f"✓ 采集图像: {results['images_collected']} 张")
    else:
        logging.warning("✗ 图像目录不存在")
    
    # 检查进度文件
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            logging.info(f"✓ 爬取进度: 已爬到第 {progress.get('last_page', 'N/A')} 页")
            logging.info(f"  已处理文章: {len(progress.get('processed_article_urls', []))} 篇")
            logging.info(f"  最后更新: {progress.get('last_update_time', 'N/A')}")
            results["last_page"] = progress.get('last_page', 1)
            results["processed_articles"] = len(progress.get('processed_article_urls', []))
        except Exception as e:
            logging.warning(f"✗ 无法读取进度文件: {e}")
    
    # 生成简要报告
    logging.info("\n" + "="*60)
    logging.info("采集统计摘要")
    logging.info("="*60)
    
    for key, value in results.items():
        logging.info(f"{key}: {value}")
    
    return results


def generate_quick_stats():
    """生成快速统计"""
    metadata_file = "scir_dataset/metadata.json"
    
    if not os.path.exists(metadata_file):
        logging.warning("元数据文件不存在，无法生成统计")
        return
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 统计学科分布
        subject_count = {}
        dimension_count = {}
        
        for item in metadata:
            subjects = item.get("subjects", [])
            for subject in subjects:
                subject_count[subject] = subject_count.get(subject, 0) + 1
            
            # 统计图像尺寸范围
            width = item.get("image_width", 0)
            height = item.get("image_height", 0)
            
            if width and height:
                if width < 500:
                    size_cat = "小图 (<500px)"
                elif width < 1000:
                    size_cat = "中图 (500-1000px)"
                else:
                    size_cat = "大图 (>1000px)"
                
                dimension_count[size_cat] = dimension_count.get(size_cat, 0) + 1
        
        # 输出统计
        logging.info("\n" + "="*60)
        logging.info("学科分布 (Top 10)")
        logging.info("="*60)
        
        for subject, count in sorted(subject_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            logging.info(f"  {subject}: {count}")
        
        logging.info("\n" + "="*60)
        logging.info("图像尺寸分布")
        logging.info("="*60)
        
        for size_cat, count in sorted(dimension_count.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {size_cat}: {count}")
        
        # 保存统计到文件
        stats = {
            "total_images": len(metadata),
            "subject_distribution": subject_count,
            "dimension_distribution": dimension_count,
            "generated_at": datetime.now().isoformat()
        }
        
        stats_file = "scir_dataset/quick_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logging.info(f"\n统计报告已保存至: {stats_file}")
        
    except Exception as e:
        logging.error(f"生成统计失败: {e}")


def main():
    """主流程"""
    print("\n" + "="*60)
    print("科学图像数据集自动化采集流水线")
    print("="*60)
    print("\n任务目标:")
    print(" 自动从Nature抓取论文图像")
    print("\n")
    
    # 步骤1: 检查依赖
    logging.info("步骤 1/4: 检查依赖包...")
    if not check_requirements():
        logging.error("依赖检查失败，请安装缺失的包")
        return
    
    # 步骤2: 运行采集
    logging.info("步骤 2/4: 开始数据采集...")
    success = run_collection()
    
    if not success:
        logging.error("数据采集失败")
        return
    
    # 步骤3: 验证结果
    logging.info("\n步骤 3/4: 验证采集结果...")
    results = verify_results()
    
    # 步骤4: 生成统计
    logging.info("\n步骤 4/4: 生成统计报告...")
    generate_quick_stats()
    
    # 最终总结
    logging.info("\n" + "="*60)
    logging.info("流水线执行完成")
    logging.info("="*60)
    
    if results["images_collected"] >= 20000:
        logging.info(f"✓ 已达成目标：采集了 {results['images_collected']} 张图片")
    else:
        logging.info(f"⚠ 采集了 {results['images_collected']} 张图片")
        logging.info(f"  距离目标20000张还差 {20000 - results['images_collected']} 张")
        logging.info("  提示：可以重新运行程序继续采集")
    
    logging.info(f"\n详细日志已保存至: collection.log")
    logging.info(f"数据目录: scir_dataset/")
    logging.info(f"  - images/        原始图像")
    logging.info(f"  - metadata.json  完整元数据")
    
    print("\n采集完成！请查看 scir_dataset/ 目录")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n\n用户中断执行")
    except Exception as e:
        logging.error(f"\n执行出错: {e}")
        import traceback
        logging.error(traceback.format_exc())

