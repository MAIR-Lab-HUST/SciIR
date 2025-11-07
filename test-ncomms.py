import os
import re
import json
import time
import copy
import logging
import string
from io import BytesIO
from typing import Dict, List, Tuple
import concurrent.futures
import threading

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from PIL import Image
from bs4 import BeautifulSoup, NavigableString
from pylatexenc.latex2text import LatexNodes2Text

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
latex_converter = LatexNodes2Text()

# 网络请求配置
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/128.0.6613.119 Safari/537.36'
}

MAX_RETRY = 3
RETRY_DELAY = 3  # 秒

# 数据采集配置
BASE_URL = "https://www.nature.com"
ARTICLE_LIST_URL_TEMPLATE = (
    "https://www.nature.com/ncomms/articles?searchType=journalSearch&sort=PubDate&type=article&page={page}"
)
START_PAGE = 1  # 设置检索页面起始页（当没有进度文件时使用此值）
TARGET_IMAGE_COUNT = 20000
MAX_FIGURES_PER_ARTICLE = 100
MAX_ARTICLE_PAGES = 200
REQUEST_DELAY = 0.8
SUBIMAGE_DIR_NAME = "subimages"
OVERLAP_SUPPRESSION = 0.2

# 多线程配置
MAX_ARTICLE_WORKERS = 12  # 并行处理文章的线程数
MAX_IMAGE_WORKERS = 25  # 并行下载图片的线程数

ResamplingMode = getattr(Image, "Resampling", Image)
RESIZE_FILTER = getattr(ResamplingMode, "LANCZOS", Image.LANCZOS)


class ScrapingError(Exception):
    """自定义异常类，用于标记采集流程中的致命错误."""


def make_dirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def normalize_url_path(path: str) -> str:
    return path.replace('\\', '/')


def generate_article_urls(session: requests.Session, start_page: int = 1, page_count: int = 50) -> List[str]:
    """批量生成文章URL列表，从指定页码开始，抓取指定数量的页面。"""
    article_urls: List[str] = []
    for page in range(start_page, start_page + page_count):
        list_url = ARTICLE_LIST_URL_TEMPLATE.format(page=page)
        logging.info(f"抓取文章列表页面: {list_url}")
        try:
            response = session.get(list_url, headers=DEFAULT_HEADERS, timeout=20)
            if response.status_code == 404:
                logging.info(f"文章列表页面 {page} 不存在，停止分页。")
                break
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"抓取文章列表失败: {e}")
            continue

        matches = re.findall(r'href="(/articles/s\d[^"#?]+)"', response.text)
        if not matches:
            logging.info(f"未在页面 {page} 找到更多文章链接，停止分页。")
            break

        for m in matches:
            full_url = BASE_URL + m
            if full_url not in article_urls:
                article_urls.append(full_url)

        time.sleep(REQUEST_DELAY)

    logging.info(f"本次收集文章链接 {len(article_urls)} 个，从页面 {start_page} 到 {start_page + page_count - 1}。")
    return article_urls


def calculate_overlap(box_a: dict, box_b: dict) -> float:
    ax1, ay1 = box_a['x'], box_a['y']
    ax2, ay2 = ax1 + box_a['w'], ay1 + box_a['h']
    bx1, by1 = box_b['x'], box_b['y']
    bx2, by2 = bx1 + box_b['w'], by1 + box_b['h']

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    if inter_w == 0 or inter_h == 0:
        return 0.0

    inter_area = inter_w * inter_h
    min_area = min(box_a['area'], box_b['area'])
    if min_area <= 0:
        return 0.0
    return inter_area / float(min_area)


def suppress_overlaps(boxes: List[dict]) -> List[dict]:
    suppressed: List[dict] = []
    for box in sorted(boxes, key=lambda b: b['area'], reverse=True):
        if all(calculate_overlap(box, keep) <= OVERLAP_SUPPRESSION for keep in suppressed):
            suppressed.append(box)
    return suppressed


def sort_boxes_by_reading_order(boxes: List[dict]) -> List[dict]:
    if not boxes:
        return []

    heights = [box['h'] for box in boxes]
    median_height = float(np.median(heights)) if heights else 0
    row_threshold = max(int(median_height * 0.6), 30)

    boxes_sorted = sorted(boxes, key=lambda b: b['y'])
    rows: List[dict] = []
    for box in boxes_sorted:
        placed = False
        for row in rows:
            if abs(box['y'] - row['ref_y']) <= row_threshold:
                row['boxes'].append(box)
                row['ref_y'] = min(row['ref_y'], box['y'])
                placed = True
                break
        if not placed:
            rows.append({'ref_y': box['y'], 'boxes': [box]})

    rows.sort(key=lambda r: r['ref_y'])
    ordered: List[dict] = []
    for row in rows:
        row['boxes'].sort(key=lambda b: b['x'])
        ordered.extend(row['boxes'])

    return ordered


def generate_subfigure_label(index: int, existing_labels: List[str], used_labels: set) -> str:
    if index < len(existing_labels):
        label = existing_labels[index]
    else:
        label = None
        for candidate in string.ascii_lowercase:
            if candidate not in used_labels and candidate not in existing_labels:
                label = candidate
                break
        if label is None:
            label = f"auto_{index + 1}"

    used_labels.add(label)
    return label


def save_subfigure_crops(image_path: str, boxes: List[dict], base_output_dir: str, image_id: str,
                         sub_captions: dict | None = None) -> List[dict]:
    if not boxes:
        return []

    caption_map = {k.lower(): v for k, v in (sub_captions or {}).items()}
    caption_labels = sorted(caption_map.keys())
    used_labels: set = set()

    subimage_dir = os.path.join(base_output_dir, SUBIMAGE_DIR_NAME, image_id)
    make_dirs(subimage_dir)

    segments: List[dict] = []

    with Image.open(image_path) as img:
        img = img.convert('RGB')
        for idx, box in enumerate(boxes):
            x1 = max(0, box['x'])
            y1 = max(0, box['y'])
            x2 = min(img.width, x1 + box['w'])
            y2 = min(img.height, y1 + box['h'])

            if x2 <= x1 or y2 <= y1:
                continue

            label = generate_subfigure_label(idx, caption_labels, used_labels)
            sub_id = f"{image_id}_{label}"
            sub_filename = f"{sub_id}.png"
            sub_path = os.path.join(subimage_dir, sub_filename)

            crop = img.crop((x1, y1, x2, y2))
            crop.save(sub_path, "PNG", optimize=True)

            segments.append({
                'sub_id': sub_id,
                'label': label,
                'bbox': {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1},
                'local_path': normalize_url_path(sub_path),
                'width': x2 - x1,
                'height': y2 - y1,
                'aspect_ratio': round((x2 - x1) / (y2 - y1), 4) if (y2 - y1) else None,
            })

    return segments


def process_math_elements(soup_element):
    """
    处理HTML元素中的数学公式，将MathML或LaTeX转换为纯文本。

    Args:
        soup_element: BeautifulSoup元素

    Returns:
        处理后的文本字符串
    """
    if not soup_element:
        return ""

    # 使用深拷贝来避免修改原始元素
    element = copy.deepcopy(soup_element)

    # 处理MathML元素
    math_elements = element.find_all(['math', 'mml:math'])
    for math_elem in math_elements:
        # 尝试提取LaTeX格式（通常在annotation元素中）
        latex_annotation = math_elem.find('annotation', {'encoding': 'application/x-tex'})
        if not latex_annotation:
            latex_annotation = math_elem.find('mml:annotation', {'encoding': 'application/x-tex'})

        if latex_annotation and latex_annotation.string:
            try:
                # 使用pylatexenc转换LaTeX为纯文本
                plain_text = latex_converter.latex_to_text(latex_annotation.string.strip())
                # 创建新的文本节点替换math元素
                math_elem.replace_with(NavigableString(f" {plain_text} "))
            except Exception as e:
                logging.debug(f"转换LaTeX失败: {e}")
                # 如果转换失败，尝试提取可见文本
                math_text = math_elem.get_text(strip=True)
                if math_text:
                    math_elem.replace_with(NavigableString(f" {math_text} "))
                else:
                    math_elem.replace_with(NavigableString(" [数学公式] "))
        else:
            # 没有LaTeX注释，尝试提取纯文本
            math_text = math_elem.get_text(strip=True)
            if math_text:
                math_elem.replace_with(NavigableString(f" {math_text} "))
            else:
                math_elem.replace_with(NavigableString(" [数学公式] "))

    # 处理span.mathjax-tex或其他包含LaTeX的元素
    latex_spans = element.find_all('span', class_='mathjax-tex')
    for span in latex_spans:
        if span.string:
            try:
                plain_text = latex_converter.latex_to_text(span.string.strip())
                span.replace_with(NavigableString(f" {plain_text} "))
            except Exception as e:
                logging.debug(f"转换LaTeX span失败: {e}")
                span.replace_with(NavigableString(f" {span.get_text(strip=True)} "))

    # 处理script标签中的数学内容（MathJax等）
    script_tags = element.find_all('script', type=['math/tex', 'math/tex; mode=display'])
    for script in script_tags:
        if script.string:
            try:
                plain_text = latex_converter.latex_to_text(script.string.strip())
                script.replace_with(NavigableString(f" {plain_text} "))
            except Exception as e:
                logging.debug(f"转换script中的LaTeX失败: {e}")
                script.replace_with(NavigableString(" [数学公式] "))

    # 获取处理后的文本
    text = element.get_text()

    # 处理可能存在的行内LaTeX表达式 (例如 $..$ 或 $$..$$)
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: convert_latex_match(m.group(1), display=True), text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', lambda m: convert_latex_match(m.group(1), display=False), text)

    # 清理多余的空白
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def convert_latex_match(latex_str, display=False):
    """
    转换单个LaTeX匹配项为纯文本。

    Args:
        latex_str: LaTeX字符串
        display: 是否为显示模式的公式

    Returns:
        转换后的纯文本
    """
    try:
        plain_text = latex_converter.latex_to_text(latex_str.strip())
        # 如果是显示模式，在前后添加换行
        if display:
            return f"\n{plain_text}\n"
        return f" {plain_text} "
    except Exception as e:
        logging.debug(f"转换LaTeX '{latex_str}' 失败: {e}")
        return f" {latex_str} "


def clean_figure_title(title: str) -> str:
    """
    清理图片标题，移除前缀如"Fig. 4:"、"Extended Data Fig. 2:"等

    Args:
        title: 原始标题文本

    Returns:
        清理后的标题
    """
    # 移除常见的图片前缀模式
    patterns = [
        r'^Fig\.\s*\d+[a-z]?\s*[:：]\s*',  # Fig. 1: 或 Fig. 1a:
        r'^Figure\s*\d+[a-z]?\s*[:：]\s*',  # Figure 1:
        r'^Extended\s+Data\s+Fig\.\s*\d+[a-z]?\s*[:：]\s*',  # Extended Data Fig. 1:
        r'^Supplementary\s+Fig\.\s*\d+[a-z]?\s*[:：]\s*',  # Supplementary Fig. 1:
        r'^Supplementary\s+Figure\s*\d+[a-z]?\s*[:：]\s*',  # Supplementary Figure 1:
    ]

    cleaned_title = title
    for pattern in patterns:
        cleaned_title = re.sub(pattern, '', cleaned_title, flags=re.IGNORECASE)

    return cleaned_title.strip()


def extract_sub_captions(description_html: str) -> dict:
    """
    使用BeautifulSoup和正则表达式从详细描述中提取子图注，处理数学公式和共用图注。
    """
    if not description_html:
        return {}

    try:
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(description_html, 'html.parser')

        # 先处理数学公式，获取纯文本版本
        text = process_math_elements(soup)

        sub_captions = {}

        # 查找所有的粗体标签
        bold_tags = soup.find_all('b')

        # 构建一个列表来存储所有的字母标记及其位置
        letter_markers = []

        for i, bold_tag in enumerate(bold_tags):
            bold_text = bold_tag.get_text(strip=True).lower()
            # 检查是否是单个字母
            if len(bold_text) == 1 and bold_text.isalpha():
                # 检查这个字母是否是有效的子图标记
                if is_valid_subfigure_marker(bold_tag):
                    # 获取这个标签在父元素中的位置
                    parent = bold_tag.parent
                    if parent:
                        # 获取标签后的所有内容
                        position = 0
                        for j, child in enumerate(parent.children):
                            if child == bold_tag:
                                position = j
                                break

                        letter_markers.append({
                            'letter': bold_text,
                            'tag': bold_tag,
                            'parent': parent,
                            'position': position,
                            'index': i
                        })

        # 分析字母标记，识别共享描述的情况
        i = 0
        while i < len(letter_markers):
            current_marker = letter_markers[i]
            shared_letters = [current_marker['letter']]

            # 检查后续的标记是否紧跟着（中间只有逗号和空格）
            j = i + 1
            while j < len(letter_markers):
                next_marker = letter_markers[j]

                # 检查两个标记之间的内容
                if current_marker['parent'] == next_marker['parent']:
                    # 获取两个标记之间的内容
                    between_content = ""
                    for child in list(current_marker['parent'].children)[
                                 current_marker['position'] + 1:next_marker['position']]:
                        if isinstance(child, NavigableString):
                            between_content += str(child)
                        else:
                            between_content += child.get_text()

                    # 检查中间内容是否只包含逗号和空格
                    between_content = between_content.strip()
                    if between_content in [',', '，', ', ', '， ', '、']:
                        # 这是一个共享描述的标记
                        shared_letters.append(next_marker['letter'])
                        j += 1
                    else:
                        break
                else:
                    break

            # 获取这组字母后面的描述内容
            last_marker = letter_markers[j - 1] if j > i else current_marker
            content = extract_content_after_marker(last_marker['tag'], last_marker['parent'], letter_markers, j)

            # 清理内容
            content = clean_caption_content(content)

            # 为所有共享的字母分配相同的描述
            if content and len(content) > 5:
                for letter in shared_letters:
                    sub_captions[letter] = content
                logging.debug(f"找到共享描述: {shared_letters} -> {content[:50]}...")

            # 移动到下一组
            i = j

        # 如果使用粗体标签方法没有找到内容，尝试纯文本方法
        if not sub_captions:
            try:
                sub_captions = extract_from_plain_text(text)
            except re.error as e:
                # 正则表达式错误，静默处理
                logging.debug(f"纯文本提取方法失败: {e}")
                pass

        # 处理字母范围（如 a-c）
        sub_captions = handle_letter_ranges(sub_captions, text)

        if sub_captions:
            logging.info(f"从描述中提取了 {len(sub_captions)} 个子图注: {list(sorted(sub_captions.keys()))}")

        return sub_captions

    except Exception as e:
        logging.debug(f"提取子图注时发生错误: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return {}


def is_valid_subfigure_marker(bold_tag) -> bool:
    """
    判断一个粗体字母标签是否是有效的子图标记。

    有效的子图标记应该：
    1. 在句子开头，或
    2. 紧跟在句号、换行等分隔符之后，或
    3. 前面有特定的标点符号（如逗号，但前面应该是另一个子图标记）

    无效的情况：
    - 在句子中间（如 "states in a with"）
    """
    parent = bold_tag.parent
    if not parent:
        return False

    # 获取该标签之前的所有文本内容
    preceding_text = ""
    for sibling in bold_tag.previous_siblings:
        if isinstance(sibling, NavigableString):
            preceding_text = str(sibling) + preceding_text
        else:
            preceding_text = sibling.get_text() + preceding_text

    # 如果之前没有文本，说明是开头，是有效的
    if not preceding_text.strip():
        return True

    # 获取紧邻的前置文本（最后30个字符）
    preceding_text = preceding_text.strip()
    last_chars = preceding_text[-30:] if len(preceding_text) > 30 else preceding_text

    # 检查是否在句子中间（前面有小写字母或特定单词）
    # 如果前面是 "in ", "of ", "to ", "from ", "with ", "by " 等介词 + 空格，很可能是句子中间
    mid_sentence_patterns = [
        r'\bin\s*$',
        r'\bof\s*$',
        r'\bto\s*$',
        r'\bfrom\s*$',
        r'\bwith\s*$',
        r'\bby\s*$',
        r'\bat\s*$',
        r'\bon\s*$',
        r'\bfor\s*$',
        r'\bas\s*$',
        r'\band\s*$',
        r'\bor\s*$',
        r'\bthe\s*$',
        r'\ba\s*$',
        r'\ban\s*$',
    ]

    for pattern in mid_sentence_patterns:
        if re.search(pattern, last_chars, re.IGNORECASE):
            return False

    # 检查后面的内容
    following_text = ""
    for sibling in bold_tag.next_siblings:
        if isinstance(sibling, NavigableString):
            following_text += str(sibling)
        else:
            following_text += sibling.get_text()
        # 只需要检查紧邻的几个字符
        if len(following_text) > 20:
            break

    following_text = following_text.lstrip()

    # 有效的子图标记后面通常跟着：
    # 1. 逗号（表示多个子图）
    # 2. 冒号或句号（表示描述开始）
    # 3. 逗号 + 空格 + 另一个字母（共享描述）
    valid_following_patterns = [
        r'^[,，]',  # 逗号开头
        r'^[.:：。]',  # 冒号或句号开头
        r'^$',  # 结尾
    ]

    for pattern in valid_following_patterns:
        if re.match(pattern, following_text):
            return True

    # 检查前面是否是句子分隔符
    sentence_end_patterns = [
        r'[.。!！?？]\s*$',  # 句号、感叹号、问号
        r'^\s*$',  # 开头
        r'\n\s*$',  # 换行
    ]

    for pattern in sentence_end_patterns:
        if re.search(pattern, preceding_text):
            return True

    # 检查是否前面是另一个子图标记（通过逗号连接）
    # 例如："a, b" 中的 b
    if re.search(r'[,，]\s*$', last_chars):
        # 再往前看是否有单个字母
        if re.search(r'\b[a-z]\s*[,，]\s*$', preceding_text, re.IGNORECASE):
            return True

    # 默认：如果前面的文本很短（少于5个字符），可能是有效的
    if len(preceding_text) < 5:
        return True

    # 其他情况视为句子中间的字母，无效
    return False


def extract_content_after_marker(marker_tag, parent, all_markers, next_marker_index):
    """
    提取标记后的内容，直到下一个有效的子图标记或段落结束。
    """
    content_parts = []

    # 获取标记在父元素中的位置
    marker_position = -1
    for i, child in enumerate(parent.children):
        if child == marker_tag:
            marker_position = i
            break

    if marker_position == -1:
        return ""

    # 收集标记后的内容
    for child in list(parent.children)[marker_position + 1:]:
        # 检查是否遇到了下一个有效的子图标记
        is_next_valid_marker = False
        if next_marker_index < len(all_markers):
            for future_marker in all_markers[next_marker_index:]:
                if child == future_marker['tag']:
                    is_next_valid_marker = True
                    break

        if is_next_valid_marker:
            break

        if isinstance(child, NavigableString):
            text = str(child).strip()
            # 跳过单独的逗号或冒号（在字母标记后）
            if text and text not in [',', '，', ':', '：', '、']:
                content_parts.append(text)
        elif child.name == 'b':
            # 检查是否是有效的子图标记
            child_text = child.get_text(strip=True).lower()
            if len(child_text) == 1 and child_text.isalpha():
                # 这是一个粗体字母，但检查它是否是有效的子图标记
                if is_valid_subfigure_marker(child):
                    # 这是一个有效的子图标记，停止收集
                    break
                else:
                    # 这是句子中间的字母引用，保留其内容
                    content_parts.append(child.get_text())
            else:
                # 不是单个字母的粗体，保留其内容
                content_parts.append(child.get_text())
        else:
            # 其他标签，提取文本
            content_parts.append(child.get_text())

    return ' '.join(content_parts)


def clean_caption_content(content):
    """
    清理子图注内容。
    """
    if not content:
        return ""

    # 移除开头的标点符号
    content = re.sub(r'^[,，:：、\s]+', '', content)

    # 规范化空白
    content = re.sub(r'\s+', ' ', content)

    # 移除末尾的句号前的多余空格
    content = re.sub(r'\s+\.', '.', content)

    return content.strip()


def extract_from_plain_text(text):
    """
    从纯文本中提取子图注（备用方法）。
    """
    sub_captions = {}

    # 简单的模式匹配
    pattern = re.compile(
        r'(?:^|\.\s+|\n)\s*'  # 句子开始
        r'\$?([a-z])$?'  # 字母
        r'\s*[,.:]\s*'  # 分隔符
        r'([^.]+?)'  # 内容
        r'(?=\s*$?[a-z]$?[,.:]|$)',  # 前瞻：下一个标记或结束
        re.IGNORECASE | re.MULTILINE
    )

    for match in pattern.finditer(text):
        letter = match.group(1).lower()
        content = match.group(2).strip()

        if content and len(content) > 5:
            sub_captions[letter] = content

    return sub_captions


def handle_letter_ranges(sub_captions, text):
    """
    处理字母范围表示法（如 a-c）。
    """
    # 在文本中查找范围模式
    range_pattern = re.compile(r'\b([a-z])\s*[-–—]\s*([a-z])\b', re.IGNORECASE)

    for match in range_pattern.finditer(text):
        start_letter = match.group(1).lower()
        end_letter = match.group(2).lower()

        # 如果起始字母有描述，扩展到范围内的其他字母
        if start_letter in sub_captions:
            content = sub_captions[start_letter]
            start_ord = ord(start_letter)
            end_ord = ord(end_letter)

            if start_ord < end_ord:
                for ord_val in range(start_ord + 1, end_ord + 1):
                    letter = chr(ord_val)
                    if letter not in sub_captions:
                        sub_captions[letter] = content

    return sub_captions


def parse_letter_combination(letters_str: str) -> list:
    """
    解析字母组合字符串，支持多种格式。

    Args:
        letters_str: 字母字符串，如 "a", "b, c", "a-c"

    Returns:
        字母列表
    """
    letters = []
    letters_str = letters_str.lower().strip()

    # 处理范围（如 a-c）
    if '-' in letters_str or '–' in letters_str or '—' in letters_str:
        range_pattern = re.compile(r'([a-z])\s*[-–—]\s*([a-z])')
        range_match = range_pattern.search(letters_str)

        if range_match:
            start_letter = range_match.group(1)
            end_letter = range_match.group(2)
            start_ord = ord(start_letter)
            end_ord = ord(end_letter)
            if start_ord <= end_ord:
                for ord_val in range(start_ord, end_ord + 1):
                    letters.append(chr(ord_val))
                return letters

    # 处理逗号分隔的列表（如 b, c）
    if ',' in letters_str or '，' in letters_str:
        parts = re.split(r'[,，]\s*', letters_str)
        for part in parts:
            part = part.strip()
            if part and part.isalpha() and len(part) == 1:
                letters.append(part)
        return letters

    # 单个字母
    if letters_str.isalpha() and len(letters_str) == 1:
        return [letters_str]

    # 如果没有解析出任何字母，尝试提取所有单个字母
    single_letters = re.findall(r'[a-z]', letters_str)
    return single_letters


def scrape_article_text(article_url: str, session: requests.Session | None = None) -> dict:
    """
    从单个Nature文章页面抓取标题、摘要、正文和引用信息，自动处理数学公式。
    """
    logging.info(f"开始从 {article_url} 抓取文本内容...")
    text_data = {"title": "", "abstract": "", "body": "", "source_citation": ""}  # 添加source_citation字段
    try:
        sess = session or requests.Session()
        for attempt in range(MAX_RETRY):
            try:
                response = sess.get(article_url, headers=DEFAULT_HEADERS, timeout=20)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                logging.warning(f"请求文章 {article_url} 失败 (尝试 {attempt + 1}/{MAX_RETRY}): {e}")
                if attempt + 1 == MAX_RETRY:
                    raise
                time.sleep(RETRY_DELAY)
        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. 抓取标题（包含数学公式处理）
        title_tag = soup.select_one('h1.c-article-title')
        if title_tag:
            # 移除可能存在的额外标签（但保留数学元素）
            for span in title_tag.find_all('span'):
                # 检查是否是与数学相关的span
                if span.get('class'):
                    classes = ' '.join(span.get('class'))
                    if 'mathjax' in classes.lower() or 'math' in classes.lower():
                        continue
                # 如果不是数学相关的，且没有数学子元素，则展开
                if not span.find_all(['math', 'mml:math', 'script']):
                    span.unwrap()

            # 处理数学公式并获取文本
            text_data["title"] = process_math_elements(title_tag)
            logging.info(f"成功抓取标题: {text_data['title']}")
        else:
            logging.warning("未找到文章标题。")

        # 2. 抓取摘要（包含数学公式处理）
        abstract_section = soup.select_one('section[aria-labelledby="Abs1"]')
        if not abstract_section:
            # 尝试其他可能的摘要选择器
            abstract_section = soup.select_one('div.c-article-section__content')

        if abstract_section:
            # 提取所有段落并处理数学公式
            abstract_paragraphs = abstract_section.select('p')
            if not abstract_paragraphs:
                # 如果没有找到p标签，尝试直接获取内容
                processed_text = process_math_elements(abstract_section)
                if processed_text:
                    text_data["abstract"] = processed_text
            else:
                abstract_text_parts = []
                for p in abstract_paragraphs:
                    processed_text = process_math_elements(p)
                    if processed_text:
                        abstract_text_parts.append(processed_text)

                text_data["abstract"] = "\n\n".join(abstract_text_parts)

            if text_data["abstract"]:
                logging.info(f"成功抓取摘要 (长度: {len(text_data['abstract'])})")
        else:
            logging.warning("未找到文章摘要。")

        # 3. 抓取正文（包含数学公式处理）
        main_content_div = soup.select_one('div.main-content')
        if not main_content_div:
            # 尝试其他可能的正文容器
            main_content_div = soup.select_one('div.c-article-body')

        if main_content_div:
            sections = main_content_div.find_all('section')
            if not sections:
                # 如果没有section，尝试直接获取段落
                paragraphs = main_content_div.find_all('p')
            else:
                paragraphs = []
                for section in sections:
                    section_title = section.get('data-title', '') or section.get('aria-label', '')
                    # 可选：添加章节标题
                    if section_title and section_title not in ['Abstract', 'References']:
                        paragraphs.append(f"## {section_title}")

                    # 提取该section内的所有段落
                    section_paragraphs = section.select('p')
                    paragraphs.extend(section_paragraphs)

            main_text_parts = []
            for p in paragraphs:
                if isinstance(p, str):
                    main_text_parts.append(p)
                else:
                    processed_text = process_math_elements(p)
                    if processed_text:
                        main_text_parts.append(processed_text)

            text_data["body"] = "\n\n".join(main_text_parts)
            if text_data["body"]:
                logging.info(f"成功抓取正文 (长度: {len(text_data['body'])})")
        else:
            logging.warning("未找到文章正文内容。")

        # 4. 抓取引用信息 (新增部分)
        citation_tag = soup.select_one('p.c-bibliographic-information__citation')
        if citation_tag:
            # 处理数学公式并获取纯文本引用信息
            citation_text = process_math_elements(citation_tag)
            if citation_text:
                # 清理引用文本中的多余空格和换行
                citation_text = re.sub(r'\s+', ' ', citation_text).strip()
                text_data["source_citation"] = citation_text
                logging.info(f"成功抓取引用信息: {citation_text[:100]}...")
        else:
            # 尝试备用选择器
            cite_section = soup.find('h3', id='citeas')
            if cite_section:
                parent = cite_section.parent
                if parent:
                    citation_p = parent.find('p', class_=re.compile('citation'))
                    if citation_p:
                        citation_text = process_math_elements(citation_p)
                        if citation_text:
                            citation_text = re.sub(r'\s+', ' ', citation_text).strip()
                            text_data["source_citation"] = citation_text
                            logging.info(f"成功抓取引用信息: {citation_text[:100]}...")

        if not text_data["source_citation"]:
            logging.warning("未找到文章引用信息。")

        return text_data

    except requests.exceptions.RequestException as e:
        logging.error(f"抓取文章文本 {article_url} 时发生网络错误: {e}")
        return text_data
    except Exception as e:
        logging.error(f"处理文章文本时发生错误: {e}")
        return text_data


def scrape_article_figures(article_url: str, max_figures: int = 50, session: requests.Session | None = None) -> list:
    """
    从单个Nature Communications文章逐个抓取图像URL和图注，自动处理数学公式。
    """
    figures_data = []
    sess = session or requests.Session()

    for figure_num in range(1, max_figures + 1):
        figure_url = f"{article_url}/figures/{figure_num}"
        try:
            for attempt in range(MAX_RETRY):
                try:
                    response = sess.get(figure_url, headers=DEFAULT_HEADERS, timeout=15)
                    if response.status_code == 404:
                        logging.info(f"图片 {figure_num} 不存在 (404)，停止抓取。")
                        break
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    logging.warning(
                        f"请求图片 {figure_url} 失败 (尝试 {attempt + 1}/{MAX_RETRY}): {e}"
                    )
                    if attempt + 1 == MAX_RETRY:
                        raise
                    time.sleep(RETRY_DELAY)
            else:
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # 查找图像标签
            image_tag = soup.select_one('figure[data-test="figure"] picture img')
            if not image_tag:
                image_tag = soup.select_one('img.c-article-figure__image')

            # 查找标题
            title_tag = soup.select_one('h1.c-article-satellite-title[data-test="top-caption"]')
            if not title_tag:
                title_tag = soup.select_one('h1.c-article-figure__caption')

            # 查找详细描述
            description_tag = soup.select_one('div.c-article-figure-description')
            if not description_tag:
                description_tag = soup.select_one('div.c-article-figure__description')

            if not image_tag:
                logging.info(f"图片 {figure_num} 页面没有找到图像，停止抓取。")
                break

            # 获取并处理图像URL
            image_url = image_tag.get('src')
            if image_url and image_url.startswith('//'):
                image_url = 'https:' + image_url
            elif image_url and not image_url.startswith('http'):
                from urllib.parse import urljoin
                image_url = urljoin(figure_url, image_url)

            # 处理标题文本（包含数学公式）
            title_text = ""
            if title_tag:
                # 移除内部的无关span标签
                for span in title_tag.find_all('span'):
                    classes = span.get('class', [])
                    if isinstance(classes, str):
                        classes = [classes]
                    classes_str = ' '.join(classes).lower()
                    if 'mathjax' not in classes_str and 'math' not in classes_str:
                        if not span.find_all(['math', 'mml:math', 'script']):
                            span.unwrap()

                title_text = process_math_elements(title_tag)
                # 清理标题前缀
                title_text = clean_figure_title(title_text)

            # 保存原始HTML用于子图解析
            description_html = ""
            description_text = ""  # 添加纯文本描述
            if description_tag:
                description_html = str(description_tag)
                # 处理数学公式并获取纯文本描述
                description_text = process_math_elements(description_tag)

            if image_url and title_text:
                figures_data.append({
                    "figure_index": figure_num,
                    "image_url": image_url,
                    "title": title_text,
                    "description": description_text,  # 添加纯文本描述
                    "description_html": description_html,  # 仅内部使用
                    "source_article_url": article_url,
                    "figure_page_url": figure_url
                })
                logging.info(f"成功抓取图片 {figure_num}: {title_text[:50] if title_text else '无标题'}...")
            else:
                logging.warning(f"图片 {figure_num} 缺少必要信息，跳过。")

        except requests.exceptions.RequestException as e:
            logging.error(f"抓取图片 {figure_num} 时发生网络错误: {e}")
            break
        except Exception as e:
            logging.error(f"处理图片 {figure_num} 时发生错误: {e}")
            continue

    logging.info(f"成功从 {article_url} 抓取了 {len(figures_data)} 张图像。")
    return figures_data


def download_and_standardize_image(image_url: str, save_path: str, max_size=4096,
                                   session: requests.Session | None = None):
    """下载图像并将其标准化为PNG格式。"""
    try:
        sess = session or requests.Session()
        for attempt in range(MAX_RETRY):
            try:
                response = sess.get(image_url, headers=DEFAULT_HEADERS, stream=True, timeout=15)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                logging.warning(f"下载图像 {image_url} 失败 (尝试 {attempt + 1}/{MAX_RETRY}): {e}")
                if attempt + 1 == MAX_RETRY:
                    raise
                time.sleep(RETRY_DELAY)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), RESIZE_FILTER)

        directory = os.path.dirname(save_path)
        if directory:
            make_dirs(directory)

        img.save(save_path, "PNG", optimize=True)

        # logging.info(f"成功下载并保存图像至 {save_path}") # 在多线程中打印过多会混乱
        return True

    except Exception as e:
        logging.error(f"下载或处理图像 {image_url} 失败: {e}")
        return False


def compile_and_save_metadata(all_metadata: list, output_file: str):
    """将所有收集和处理过的元数据编译成一个JSON文件并保存。"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=4)
    logging.info(f"所有元数据已成功归档至 {output_file}。")


def load_progress(output_dir: str) -> dict:
    """加载爬取进度信息。"""
    progress_file = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            logging.info(f"成功加载进度文件: 上次爬到第 {progress.get('last_page', START_PAGE)} 页，"
                         f"已处理 {len(progress.get('processed_article_urls', []))} 篇文章，"
                         f"已采集 {progress.get('total_images', 0)} 张图片")
            return progress
        except Exception as e:
            logging.warning(f"加载进度文件失败: {e}，将从配置的起始页开始")
    return {
        'last_page': START_PAGE,
        'processed_article_urls': [],
        'total_images': 0,
        'last_update_time': None
    }


def save_progress(output_dir: str, current_page: int, processed_article_urls: set, total_images: int):
    """保存当前爬取进度。"""
    progress_file = os.path.join(output_dir, "progress.json")
    progress = {
        'last_page': current_page,
        'processed_article_urls': list(processed_article_urls),
        'total_images': total_images,
        'last_update_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=4)
        logging.info(f"进度已保存: 当前第 {current_page} 页，已处理 {len(processed_article_urls)} 篇文章，"
                     f"已采集 {total_images} 张图片")
    except Exception as e:
        logging.error(f"保存进度失败: {e}")


def add_image_metadata(metadata: dict, image_path: str):
    """为单张图像元数据添加尺寸信息。"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        metadata['image_width'] = width
        metadata['image_height'] = height
        metadata['aspect_ratio'] = round(width / height, 4) if height else None
    except Exception as e:
        logging.warning(f"读取图像尺寸失败 {image_path}: {e}")


import re
import json
import logging
from bs4 import BeautifulSoup
from typing import List


def extract_subjects_from_article(soup: BeautifulSoup) -> List[str]:
    """从 Nature 文章页面中提取学科标签。"""
    subjects = []

    data_layer_scripts = soup.find_all("script", {"data-test": "dataLayer"})
    for script in data_layer_scripts:
        script_content = script.get_text()
        if not script_content:
            continue

        # 用正则安全提取第一个 window.dataLayer = [...] 段
        match = re.search(r'window\.dataLayer\s*=\s*(\[[\s\S]*?\]);', script_content)
        if not match:
            continue

        json_str = match.group(1)  # 只取第一个数组
        try:
            data_layer = json.loads(json_str)
        except Exception as e:
            logging.error(f"JSON解析失败: {e}")
            continue

        if not isinstance(data_layer, list) or not data_layer:
            continue

        data = data_layer[0]

        # 1️⃣ 从 content.contentInfo.subjects 提取
        subjects_str = (
            data.get("content", {})
            .get("contentInfo", {})
            .get("subjects", "")
        )
        if subjects_str:
            subjects += [s.strip() for s in subjects_str.split(",") if s.strip()]

    # 对所有提取到的学科名称进行标准化
    standardized_subjects = [standardize_subject_name(s) for s in subjects]
    # 去重
    subjects = list(dict.fromkeys(standardized_subjects))

    if subjects:
        logging.info(f"成功提取 {len(subjects)} 个学科标签: {subjects}")
    else:
        logging.warning("未能提取到学科信息")

    return subjects


def add_subject_metadata(metadata: dict, subjects: List[str]):
    """
    在元数据中添加学科标签和对应的领域。

    Args:
        metadata: 元数据字典
        subjects: 学科列表
    """
    if subjects:
        metadata['subjects'] = subjects
    else:
        metadata['subjects'] = []


def standardize_subject_name(subject: str) -> str:
    """
    标准化学科名称格式，确保与Nature定义的标准学科名称一致。

    Args:
        subject: 原始学科名称

    Returns:
        标准化后的学科名称
    """
    subject = subject.strip()

    # 定义标准学科名称映射（从各种可能的变体到标准名称）
    subject_mapping = {
        # Physical sciences
        'physics': 'Physics',
        'astronomy': 'Astronomy and planetary science',
        'astronomy and planetary science': 'Astronomy and planetary science',
        'chemistry': 'Chemistry',
        'materials science': 'Materials science',
        'mathematics': 'Mathematics and computing',
        'mathematics and computing': 'Mathematics and computing',
        'engineering': 'Engineering',
        'nanoscience': 'Nanoscience and technology',
        'nanoscience and technology': 'Nanoscience and technology',
        'nanotechnology': 'Nanoscience and technology',
        'optics': 'Optics and photonics',
        'optics and photonics': 'Optics and photonics',
        'photonics': 'Optics and photonics',
        'energy science': 'Energy science and technology',
        'energy science and technology': 'Energy science and technology',

        # Earth and environmental sciences
        'climate sciences': 'Climate sciences',
        'climate science': 'Climate sciences',
        'climate change': 'Climate sciences',
        'ecology': 'Ecology',
        'environmental sciences': 'Environmental sciences',
        'environmental science': 'Environmental sciences',
        'solid earth sciences': 'Solid Earth sciences',
        'earth sciences': 'Solid Earth sciences',
        'planetary science': 'Planetary science',
        'environmental social sciences': 'Environmental social sciences',
        'biogeochemistry': 'Biogeochemistry',
        'ocean sciences': 'Ocean sciences',
        'oceanography': 'Ocean sciences',
        'hydrology': 'Hydrology',
        'natural hazards': 'Natural hazards',
        'limnology': 'Limnology',
        'space physics': 'Space physics',

        # Biological sciences
        'genetics': 'Genetics',
        'genomics': 'Genetics',
        'microbiology': 'Microbiology',
        'neuroscience': 'Neuroscience',
        'immunology': 'Immunology',
        'evolution': 'Evolution',
        'cancer': 'Cancer',
        'oncology': 'Cancer',
        'cell biology': 'Cell biology',
        'biochemistry': 'Biochemistry',
        'molecular biology': 'Molecular biology',
        'zoology': 'Zoology',
        'developmental biology': 'Developmental biology',
        'biological techniques': 'Biological techniques',
        'structural biology': 'Structural biology',
        'physiology': 'Physiology',
        'biotechnology': 'Biotechnology',
        'computational biology': 'Computational biology and bioinformatics',
        'computational biology and bioinformatics': 'Computational biology and bioinformatics',
        'bioinformatics': 'Computational biology and bioinformatics',
        'drug discovery': 'Drug discovery',
        'stem cells': 'Stem cells',
        'stem cell': 'Stem cells',
        'plant sciences': 'Plant sciences',
        'plant science': 'Plant sciences',
        'plant biology': 'Plant sciences',
        'psychology': 'Psychology',
        'biophysics': 'Biophysics',
        'chemical biology': 'Chemical biology',
        'systems biology': 'Systems biology',

        # Health sciences
        'diseases': 'Diseases',
        'disease': 'Diseases',
        'health care': 'Health care',
        'healthcare': 'Health care',
        'medical research': 'Medical research',
        'medicine': 'Medical research',
        'anatomy': 'Anatomy',
        'pathogenesis': 'Pathogenesis',
        'biomarkers': 'Biomarkers',
        'biomarker': 'Biomarkers',
        'risk factors': 'Risk factors',
        'neurology': 'Neurology',
        'signs and symptoms': 'Signs and symptoms',
        'endocrinology': 'Endocrinology',
        'health occupations': 'Health occupations',

        # Scientific community and society
        'scientific community': 'Scientific community',
        'social sciences': 'Social sciences',
        'social science': 'Social sciences',
        'business and industry': 'Business and industry',
        'developing world': 'Developing world',
        'agriculture': 'Agriculture',
        'water resources': 'Water resources',
        'geography': 'Geography',
        'energy and society': 'Energy and society',
        'forestry': 'Forestry'
    }

    # 先尝试小写匹配
    subject_lower = subject.lower().replace('-', ' ')
    if subject_lower in subject_mapping:
        return subject_mapping[subject_lower]

    # 如果没有找到映射，检查是否已经是标准格式
    # 获取所有标准学科名称
    all_standard_subjects = set(subject_mapping.values())
    if subject in all_standard_subjects:
        return subject

    # 如果还是没有找到，返回原始值（首字母大写）
    return subject


def fetch_article_page(article_url: str, session: requests.Session | None = None) -> BeautifulSoup | None:
    """抓取文章页面并返回解析后的Soup对象。"""
    sess = session or requests.Session()
    for attempt in range(MAX_RETRY):
        try:
            response = sess.get(article_url, headers=DEFAULT_HEADERS, timeout=20)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            logging.warning(f"请求文章页面 {article_url} 失败 (尝试 {attempt + 1}/{MAX_RETRY}): {e}")
            if attempt + 1 == MAX_RETRY:
                logging.error(f"无法获取文章页面: {article_url}")
                return None
            time.sleep(RETRY_DELAY)
    return None


def check_article_license(soup: BeautifulSoup) -> bool:
    """
    检查文章是否为CC BY协议。

    Args:
        soup: 文章页面的BeautifulSoup对象

    Returns:
        bool: 如果是CC BY协议返回True，否则返回False
    """
    try:
        # 查找Rights and permissions部分
        rights_section = soup.find('section', {'data-title': 'Rights and permissions'})
        if not rights_section:
            # 尝试其他选择器
            rights_section = soup.find('section', id='rightslink-section')

        if not rights_section:
            logging.info("未找到Rights and permissions部分，跳过该文章")
            return False

        # 获取文本内容
        rights_text = rights_section.get_text(separator=' ', strip=True).lower()

        # 检查是否为开放访问
        if 'open access' not in rights_text:
            logging.info("该文章不是开放访问，跳过")
            return False

        # 定义需要排除的CC协议变体
        excluded_licenses = [
            'creative commons attribution-sharealike',
            'creative commons attribution-noncommercial-sharealike',
            'creative commons attribution-noncommercial',
            'creative commons attribution-noderivs',
            'creative commons attribution-noncommercial-noderivs',
            'cc by-sa',
            'cc by-nc',
            'cc by-nc-sa',
            'cc by-nd',
            'cc by-nc-nd',
        ]

        # 检查是否包含排除的协议
        for excluded in excluded_licenses:
            if excluded in rights_text:
                logging.info(f"文章使用受限的CC协议 ({excluded})，跳过")
                return False

        # 检查是否为CC BY 4.0协议
        # 可能的表述方式
        cc_by_patterns = [
            'creative commons attribution 4.0 international license',
            'creative commons attribution 4.0',
            'cc by 4.0',
            'licensed under a creative commons attribution 4.0'
        ]

        for pattern in cc_by_patterns:
            if pattern in rights_text:
                # 最终确认：检查链接中是否确实指向CC BY 4.0
                cc_link = rights_section.find('a', href=re.compile(r'creativecommons\.org/licenses/by/4\.0'))
                if cc_link:
                    logging.info("✓ 文章使用CC BY 4.0协议，允许爬取")
                    return True

                # 也接受没有版本号的CC BY链接
                cc_link = rights_section.find('a', href=re.compile(r'creativecommons\.org/licenses/by/(?!-)'))
                if cc_link:
                    # 确保URL不包含其他限制（如by-nc、by-sa等）
                    href = cc_link.get('href', '').lower()
                    if not any(x in href for x in ['-nc', '-sa', '-nd']):
                        logging.info("✓ 文章使用CC BY协议，允许爬取")
                        return True

        # 如果找到了rights部分但不是CC BY，记录具体原因
        if 'creative commons' in rights_text:
            logging.info("文章使用非CC BY的Creative Commons协议，跳过")
        else:
            logging.info("文章不使用CC BY协议，跳过")

        return False

    except Exception as e:
        logging.warning(f"检查文章许可证时出错: {e}")
        return False


# --- 多线程优化部分 ---

# 全局计数器和锁
image_counter = 0
counter_lock = threading.Lock()


def process_article(article_url: str, session: requests.Session) -> dict | None:
    """
    处理单个文章：检查许可、抓取文本、学科和图片元数据。
    这是一个独立的任务单元，用于线程池。
    """
    logging.info(f"检查文章: {article_url}")
    article_soup = fetch_article_page(article_url, session=session)
    if not article_soup:
        return None

    if not check_article_license(article_soup):
        logging.info(f"跳过非CC BY协议文章: {article_url}")
        return {'status': 'skipped_license'}

    logging.info(f"处理CC BY文章: {article_url}")
    subjects = extract_subjects_from_article(article_soup)
    article_text_data = scrape_article_text(article_url, session=session)
    figures = scrape_article_figures(article_url, max_figures=MAX_FIGURES_PER_ARTICLE, session=session)

    return {
        'status': 'success',
        'subjects': subjects,
        'text_data': article_text_data,
        'figures': figures
    }


def process_figure(figure_task: tuple, images_dir: str, session: requests.Session) -> dict | None:
    """
    处理单个图片：下载、保存并生成元数据。
    这是一个独立的任务单元，用于线程池。
    """
    global image_counter

    fig_data, article_text_data, subjects = figure_task

    with counter_lock:
        if image_counter >= TARGET_IMAGE_COUNT:
            return None
        current_count = image_counter
        image_counter += 1

    image_id = f"sciir_img_{current_count:06d}"
    image_filename = f"{image_id}.png"
    local_path = os.path.join(images_dir, image_filename)

    logging.info(f"下载图片 {current_count + 1}/{TARGET_IMAGE_COUNT}: {fig_data['image_url']}")
    success = download_and_standardize_image(fig_data['image_url'], local_path, session=session)
    if not success:
        # 如果下载失败，需要将计数器减回去以确保准确性
        with counter_lock:
            image_counter -= 1
        return None

    sub_captions = extract_sub_captions(fig_data.get("description_html", ""))

    metadata = {
        "image_id": image_id,
        "local_path": local_path,
        "article_title": article_text_data["title"],
        "article_abstract": article_text_data["abstract"],
        "article_body": article_text_data["body"],
        "source_citation": article_text_data.get("source_citation", ""),
        "figure_title": fig_data["title"],
        "figure_index": fig_data["figure_index"],
        "image_url": fig_data["image_url"],
        "source_article_url": fig_data["source_article_url"],
        "figure_page_url": fig_data["figure_page_url"],
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/"
    }

    if sub_captions:
        metadata["has_subfigures"] = True
        metadata["sub_captions"] = sub_captions
    else:
        metadata["has_subfigures"] = False
        metadata["figure_caption"] = fig_data.get("description", "")

    add_image_metadata(metadata, local_path)
    add_subject_metadata(metadata, subjects)

    return metadata


def main():
    """主执行函数（多线程优化版，支持动态抓取）。"""
    global image_counter
    output_dir = "scir_dataset"
    images_dir = os.path.join(output_dir, "images")
    make_dirs(images_dir)
    existing_images = [f for f in os.listdir(images_dir) if f.endswith('.png')] if os.path.exists(images_dir) else []
    start_count = len(existing_images)
    image_counter = start_count
    logging.info(f"检测到已有 {start_count} 张图片，将继续采集至 {TARGET_IMAGE_COUNT} 张")
    if start_count >= TARGET_IMAGE_COUNT:
        logging.info("目标已达成，无需采集。")
        return

    metadata_path = os.path.join(output_dir, "metadata.json")
    all_metadata = []
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
            logging.info(f"成功加载 {len(all_metadata)} 条现有元数据记录")
        except Exception as e:
            logging.warning(f"加载元数据失败: {e}，将创建新的元数据文件")

    # 加载爬取进度
    progress = load_progress(output_dir)

    try:
        with requests.Session() as session:
            # 配置连接池上限为20
            adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            # 创建已处理文章URL的集合（优先从进度文件加载，其次从元数据提取）
            processed_article_urls = set(progress.get('processed_article_urls', []))
            if not processed_article_urls:
                # 如果进度文件中没有，则从元数据中提取
                for metadata in all_metadata:
                    if 'source_article_url' in metadata:
                        processed_article_urls.add(metadata['source_article_url'])

            # 初始化动态抓取参数（从进度文件恢复，如果没有进度文件则使用配置的起始页）
            start_page = progress.get('last_page', START_PAGE)
            page_increment = 5  # 每次增量抓取5页
            total_collected_images = start_count
            total_processed_articles = len(processed_article_urls)
            cc_by_articles = 0  # 计数CC BY文章数量

            logging.info(f"开始动态抓取流程，目标: {TARGET_IMAGE_COUNT} 张图片")
            logging.info(f"从第 {start_page} 页开始继续爬取")

            while total_collected_images < TARGET_IMAGE_COUNT:
                logging.info(f"动态抓取阶段: 从页面 {start_page} 开始，抓取 {page_increment} 页")

                # 抓取下一批文章列表页面
                article_urls = generate_article_urls(session, start_page=start_page, page_count=page_increment)

                if not article_urls:
                    logging.info("未找到更多文章URL，无法继续抓取。")
                    break

                # 过滤掉已处理的文章
                new_article_urls = [url for url in article_urls if url not in processed_article_urls]
                if not new_article_urls:
                    logging.info("没有新的文章URL需要处理。")
                    start_page += page_increment
                    continue

                logging.info(
                    f"找到 {len(new_article_urls)} 篇新文章（总共 {len(article_urls)} 篇，已处理 {len(processed_article_urls)} 篇）")

                # --- 阶段一: 并行处理新文章 ---
                logging.info(f"===== 阶段一: 开始并行处理 {len(new_article_urls)} 篇新文章... =====")
                figure_tasks = []
                processed_articles = 0
                skipped_articles = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_ARTICLE_WORKERS) as executor:
                    future_to_url = {executor.submit(process_article, url, session): url for url in new_article_urls}
                    for future in concurrent.futures.as_completed(future_to_url):
                        result = future.result()
                        if result:
                            if result['status'] == 'success':
                                processed_articles += 1
                                cc_by_articles += 1
                                # 为每个图片创建一个任务元组
                                for fig in result['figures']:
                                    figure_tasks.append((fig, result['text_data'], result['subjects']))
                            elif result['status'] == 'skipped_license':
                                skipped_articles += 1

                logging.info(
                    f"阶段一完成。有效CC BY文章: {processed_articles}, 跳过: {skipped_articles}。共找到 {len(figure_tasks)} 个潜在图片。")

                # 更新已处理文章集合
                processed_article_urls.update(new_article_urls)
                total_processed_articles += len(new_article_urls)

                # --- 阶段二: 并行下载图片 ---
                if not figure_tasks:
                    logging.info("未找到任何可下载的图片，继续抓取更多文章...")
                    # 保存进度后继续
                    save_progress(output_dir, start_page, processed_article_urls, total_collected_images)
                    start_page += page_increment
                    continue

                logging.info(f"===== 阶段二: 开始并行下载图片... =====")
                new_metadata = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS) as executor:
                    # 创建一个任务列表，每个任务包含所有需要的参数
                    tasks = [(task, images_dir, session) for task in figure_tasks]
                    # 使用map来处理，它会返回一个结果的迭代器
                    results = executor.map(lambda p: process_figure(*p), tasks)
                    for metadata in results:
                        if metadata:
                            new_metadata.append(metadata)

                # 更新已收集的图片数量
                new_images_count = len(new_metadata)
                total_collected_images += new_images_count
                logging.info(
                    f"成功下载 {new_images_count} 张新图片，当前总计: {total_collected_images}/{TARGET_IMAGE_COUNT}")

                # 合并并保存元数据
                all_metadata.extend(new_metadata)
                compile_and_save_metadata(all_metadata, metadata_path)

                # 保存进度
                save_progress(output_dir, start_page, processed_article_urls, total_collected_images)

                # 检查是否达到目标
                if total_collected_images >= TARGET_IMAGE_COUNT:
                    break

                # 动态调整策略
                if cc_by_articles > 0:
                    # 计算每篇CC BY文章平均提供多少图片
                    avg_images_per_article = total_collected_images / cc_by_articles
                    # 预估还需要多少篇文章才能达到目标
                    remaining_images = TARGET_IMAGE_COUNT - total_collected_images
                    estimated_needed_articles = max(1, int(remaining_images / avg_images_per_article) + 1)
                    logging.info(f"当前每篇CC BY文章平均提供 {avg_images_per_article:.2f} 张图片")
                    logging.info(f"预估还需要 {estimated_needed_articles} 篇文章才能达到目标")

                    # 如果预估需要的文章数量较少，减少增量抓取的页数
                    if estimated_needed_articles < page_increment * 2:
                        page_increment = max(10, int(estimated_needed_articles * 0.5))
                        logging.info(f"调整抓取增量为 {page_increment} 页")

                # 继续抓取下一批
                start_page += page_increment

            # 最终保存进度
            save_progress(output_dir, start_page, processed_article_urls, total_collected_images)

            # 最终日志
            logging.info(f"\nSciIR数据集收集流程完成！")
            logging.info(f"总计: {total_collected_images} 张图像")
            logging.info(f"处理了 {total_processed_articles} 篇文章（其中 {cc_by_articles} 篇为CC BY协议）")
            if total_collected_images < TARGET_IMAGE_COUNT:
                logging.warning(f"距离目标还差: {TARGET_IMAGE_COUNT - total_collected_images} 张")

    except KeyboardInterrupt:
        # 用户中断时保存进度
        logging.info("\n检测到用户中断，正在保存进度...")
        save_progress(output_dir, start_page, processed_article_urls, total_collected_images)
        logging.info("进度已保存，下次运行将从此处继续")
        raise

    except Exception as e:
        # 发生异常时也保存进度
        logging.error(f"\n程序执行出错: {e}")
        save_progress(output_dir, start_page, processed_article_urls, total_collected_images)
        logging.info("进度已保存，下次运行将从此处继续")
        raise