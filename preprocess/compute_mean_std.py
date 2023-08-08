import pandas as pd
import os
import cv2
from PIL import Image
import numpy as np
import albumentations
from albumentations import pytorch as AT
from torchvision import transforms

classes = ["1.静息-标准", "2.静息-非标准"]  # label dictionary
# G = ["G6", "G7", "G8", "G10"]
path = "../../data/TrainSet"

img_paths = []
labels = []

for class_name in classes:
    class_dir = os.path.join(path, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.endswith('.bmp'):
            img_path = os.path.join(class_dir, file_name)
            img_paths.append(img_path)
            labels.append(class_name)



means = [0, 0, 0]
stds = [0, 0, 0]

img_num = len(img_paths)
for img_path in img_paths:
    # img = Image.open(img_path)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    # img = transforms.ToTensor()(img)
    img = AT.ToTensorV2()(image=img)["image"]
    # print(img)
    print(img.dtype)
    for i in range(3):
        means[i] += img[i, :, :].mean()
        stds[i] += img[i, :, :].std()
mean = np.array(means) / img_num
std = np.array(stds) / img_num  
print(mean, std)  
