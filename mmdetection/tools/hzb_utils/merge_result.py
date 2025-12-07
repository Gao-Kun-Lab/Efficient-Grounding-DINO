import os
import numpy as np
import re

def convert_bbox_to_quad(x1, y1, x2, y2):
    """将水平框 (x1, y1, x2, y2) 转为四点坐标"""
    return [x1, y1, x2, y1, x2, y2, x1, y2]

def transform_bbox_to_global(bbox, x_offset, y_offset):
    """将小图的目标框映射回大图坐标"""

    bbox_global = [
        bbox[0] + x_offset,
        bbox[1] + y_offset,
        bbox[2] + x_offset,
        bbox[3] + y_offset,
    ]
    x1, y1, x2, y2 = bbox_global
    return convert_bbox_to_quad(x1, y1, x2, y2)

def parse_detection_file(detection_file):
    """解析目标检测结果文件，返回全局坐标系下的检测框字典"""
    nameboxdict = {}

    with open(detection_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        splitline = line.strip().split(' ')
        subname = splitline[0]

        # 解析子图名，获取原始大图名、偏移量
        splitname = subname.split('__')
        oriname = splitname[0]  # 原始大图名称
        x_offset, y_offset = int(splitname[2]), int(splitname[3][1:])

        confidence = float(splitline[1])
        bbox = list(map(float, splitline[2:6]))  # 提取水平框 (x1, y1, x2, y2)

        # 转换为大图坐标
        global_bbox = transform_bbox_to_global(bbox, x_offset, y_offset)
        det = global_bbox + [confidence]  # 添加置信度

        if oriname not in nameboxdict:
            nameboxdict[oriname] = []
        nameboxdict[oriname].append(det)

    return nameboxdict

def py_cpu_nms(dets, thresh=0.5):
    """使用IoU进行非极大值抑制（NMS）"""
    if len(dets) == 0:
        return []

    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 4]
    y2 = dets[:, 5]
    scores = dets[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        indices = np.where(iou <= thresh)[0]
        order = order[indices + 1]

    return dets[keep]

def apply_nms_to_dict(nameboxdict, nms_thresh=0.5):
    """对每张图片的目标检测框应用NMS"""
    for imgname in nameboxdict:
        nameboxdict[imgname] = py_cpu_nms(nameboxdict[imgname], thresh=nms_thresh).tolist()
    return nameboxdict

def save_merged_results(nameboxdict, output_file):
    """将合并后的检测框写入txt文件"""
    with open(output_file, 'w') as f_out:
        for imgname, dets in nameboxdict.items():
            for det in dets:
                confidence = det[-1]
                bbox = det[:-1]  # 去掉最后的置信度
                outline = f"{imgname} {confidence} " + " ".join(map(str, bbox))
                f_out.write(outline + '\n')

if __name__ == "__main__":
    input_file = "/data1/detection_data/datasets_Sentinel2/myself/imshow_result/The ship is moored at the dock.txt"  # 替换为你的检测结果文件路径
    output_file = "/data1/detection_data/datasets_Sentinel2/myself/imshow_result/The ship is moored at the dock_merge.txt"  # 生成的大图检测结果

    nameboxdict = parse_detection_file(input_file)
    nameboxdict = apply_nms_to_dict(nameboxdict, nms_thresh=0.1)
    save_merged_results(nameboxdict, output_file)

    print(f"合并完成，NMS 处理后结果保存至 {output_file}")