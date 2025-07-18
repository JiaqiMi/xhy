"""
将Labelme标注的json文件转化成VOC格式的标签文件, txt格式
并将同名文件保存到指定路径
"""

import os
import json
from PIL import Image

# 可根据你的数据情况修改
IMAGE_EXTS = ['.jpg', '.jpeg', '.png']

# 指定路径
input_dir = './images/holes_labeled'            # 存放原始数据的路径，包含json和jpg文件
output_dir = './xhy0718-holes'                  # 新数据集label存放路径


output_label_dir = os.path.join(output_dir, 'label-all')
output_image_dir = os.path.join(output_dir, 'image-all')
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

# 分类名与ID映射（一定要你手动确认顺序）
class_name_to_id = {
    "red": 0,
    "green": 1,
    "black": 2
}


def convert_shape_to_bbox(shape):
    points = shape['points']
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height


def convert_to_yolo(size, bbox):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = bbox[0] * dw
    y = bbox[1] * dh
    w = bbox[2] * dw
    h = bbox[3] * dh
    return x, y, w, h


for file in os.listdir(input_dir):
    if file.endswith('.json'):
        json_path = os.path.join(input_dir, file)
        image_name = os.path.splitext(file)[0]

        # 寻找对应图片
        for ext in IMAGE_EXTS:
            image_path = os.path.join(input_dir, image_name + ext)
            if os.path.exists(image_path):
                break
        else:
            print(f"图片缺失: {image_name}")
            continue

        # 打开图片获取尺寸
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            
            # 将对应图片进行保存
            img.save(os.path.join(output_image_dir, image_name + ext))

        # 读取标注
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        label_lines = []
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_name_to_id:
                print(f"[跳过] 未知类别: {label}")
                continue

            bbox = convert_shape_to_bbox(shape)
            yolo_box = convert_to_yolo((img_width, img_height), bbox)
            class_id = class_name_to_id[label]
            label_lines.append(f"{class_id} {' '.join([f'{v:.6f}' for v in yolo_box])}")

        # 保存为YOLO格式
        out_label_path = os.path.join(output_label_dir, image_name + '.txt')
        with open(out_label_path, 'w') as out_file:
            out_file.write('\n'.join(label_lines))
            
        

print("✅ 转换完成，YOLO标签保存在:", output_label_dir)
