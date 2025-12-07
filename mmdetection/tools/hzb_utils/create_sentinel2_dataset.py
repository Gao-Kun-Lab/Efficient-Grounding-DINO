import os
import cv2
import numpy as np
import webcolors
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET

def closest_color(requested_color):
    # 定义基本颜色对照表
    css3_db = webcolors.CSS3_NAMES_TO_HEX  # 获取 CSS3 颜色名称和对应的 HEX 值
    rgb_colors = [webcolors.hex_to_rgb(hex) for hex in css3_db.values()]
    color_names = list(css3_db.keys())

    # 构建 KDTree 以找到最近的颜色
    kdtree = KDTree(rgb_colors)
    _, index = kdtree.query(requested_color)

    return color_names[index]

def create_color_label(img_dir, txt_dir):
    # 读取图像
    image = cv2.imread('your_image.jpg')  # 替换成你的图像路径
    height, width, _ = image.shape

    # 定义多边形区域的坐标
    polygon = np.array([
        [9926.11, 2939.46],
        [9920.49, 2942.27],
        [9913.61, 2928.52],
        [9919.23, 2925.71]
    ], dtype=np.int32)

    # 创建掩码
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # 计算区域的平均颜色
    mean_color = cv2.mean(image, mask=mask)[:3]  # 只取 BGR 颜色
    mean_color_rgb = mean_color[::-1]  # OpenCV 默认是 BGR，转换成 RGB

    # 获取最接近的颜色名称
    color_name = closest_color(mean_color_rgb)
    print(f"选定区域的颜色: {color_name}")


def create_xml(filename, database, width, height, depth, objects, output_file):
    root = ET.Element("annotation")

    ET.SubElement(root, "filename").text = filename

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = database

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(root, "segmented").text = "0"

    for obj in objects:
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "name").text = obj["name"]
        ET.SubElement(obj_elem, "pose").text = obj.get("pose", "Unspecified")

        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["bndbox"]["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["bndbox"]["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["bndbox"]["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["bndbox"]["ymax"])

        if "description" in obj:
            ET.SubElement(obj_elem, "description").text = obj["description"]

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)