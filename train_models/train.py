# 运行环境
# GPU： conda activate python38
# macOS: conda activate deep_learning

# =====  训练脚本  ===== # 
# yolo detect train model=yolov8s.pt data=./yaml/xhy0718-holes.yaml epochs=100 imgsz=640 batch=16


# =====  测试过程  ===== # 
yolo detect predict model=runs/detect/train/weights/best.pt source=/path/to/images


# =====  高级选项  ===== # 
# 冻结部分网络层：freeze=10（冻结前10层）
# 修改 anchor、IoU 等超参：编辑 hyp.yaml
# 继续训练已有模型：model=path/to/your_model.pt

