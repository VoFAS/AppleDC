#coding:utf-8
import torchvision
from ultralytics import YOLO
import torch


if __name__ == '__main__':
    # # 加载预训练模型
    # model = YOLO("yolov8n-p2.pt")
    # model.train(data='datasets/DiseaseData', epochs=300, batch=4)
    # results = model.val()
    # 加载预训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Torch version:", torch.__version__)
    print("Torch_vision version:", torchvision.__version__)
    model = YOLO("yolov8n-cls.pt")
    model.train(data='datasets/DiseaseData', epochs=300, batch=128)



