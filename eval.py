# -*- coding: utf-8 -*-
from ultralytics import YOLO
import matplotlib
import os
matplotlib.use('TkAgg')
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score
from sklearn.metrics import classification_report

# 验证结果
model = YOLO('runs/classify/train16/weights/best.pt')
# 模型标签
names = {0: 'Alternaria leaf spot', 1: 'Brown spot', 2: 'Frogeye leaf spot', 3: 'Grey spot', 4: 'Health',5:'Mosaic',6:'Powdery mildew',7:'Rust',8:'Scab'}
# 验证集路径
base_path = 'datasets/DiseaseData/val'

# 将names的key与value值互换，存入dict_names中
dict_names = {v: k for k, v in names.items()}
# 存储真实标签
real_labels = []
# 存储预测标签
pre_labels = []
# 遍历base_path下的所有文件夹，每个文件夹是一个分类
for i in os.listdir(base_path):
    label = dict_names[i]
    # 获取base_path下的所有文件夹下的所有图片
    for j in os.listdir(os.path.join(base_path, i)):
        # 获取图片的路径
        img_path = os.path.join(base_path, i, j)
        # 检测图片
        res = model.predict(img_path)[0]
        # 图片真实标签
        real_labels.append(label)
        # 图片预测标签
        pre_labels.append(res.probs.top1)
print("每个类别的精确率、召回率和F1-Score：")
print(classification_report(real_labels, pre_labels, target_names=list(names.values())))
