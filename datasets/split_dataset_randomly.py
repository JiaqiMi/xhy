import os
import random
import shutil

# 配置参数
source_image_dir = 'raw_data/XHY/images'  # 原始文件夹路径，如 './images'
source_label_dir = 'raw_data/XHY/labels'  # 原始文件夹路径，如 './images'
target_image_dir1 = 'under_water_objections/images/train'       # 第一部分输出路径
target_image_dir2 = 'under_water_objections/images/test'         # 第二部分输出路径
target_label_dir1 = 'under_water_objections/labels/train'       # 第一部分输出路径
target_label_dir2 = 'under_water_objections/labels/test'         # 第二部分输出路径
split_ratio = 0.8                  # 比例，80%用于 train，20%用于 val

# 创建输出目录
os.makedirs(target_image_dir1, exist_ok=True)
os.makedirs(target_image_dir2, exist_ok=True)
os.makedirs(target_label_dir1, exist_ok=True)
os.makedirs(target_label_dir2, exist_ok=True)

# 获取所有文件
files = [f for f in os.listdir(source_image_dir) if os.path.isfile(os.path.join(source_image_dir, f))]
random.shuffle(files)

# 划分
split_index = int(len(files) * split_ratio)
files_part1 = files[:split_index]
files_part2 = files[split_index:]

# 拷贝文件
for f in files_part1:
    shutil.copy(os.path.join(source_image_dir, f), os.path.join(target_image_dir1, f))

for f in files_part2:
    shutil.copy(os.path.join(source_image_dir, f), os.path.join(target_image_dir2, f))
print(files_part1)

# 划分同名标签文件
label_files_part1 = [x.split('.')[0] + '.txt' for x in files_part1]
label_files_part2 = [x.split('.')[0] + '.txt' for x in files_part2]

print(label_files_part1)

# 拷贝文件
for f in label_files_part1:
    print(f"copy {f} ----")
    shutil.copy(os.path.join(source_label_dir, f), os.path.join(target_label_dir1, f))
    print("successfully!")

for f in label_files_part2:
    print(f"copy {f} ----")
    shutil.copy(os.path.join(source_label_dir, f), os.path.join(target_label_dir2, f))
    print("successfully!")

print(f"✅ 数据划分完成：{len(files_part1)} 个文件到 '{target_image_dir1}'，{len(files_part2)} 个文件到 '{target_image_dir2}'")
