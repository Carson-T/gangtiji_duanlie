import cv2
from glob import glob
import os
import numpy as np
from tqdm import tqdm
from PIL import Image


def boxCropForUltrasound(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (1340, 700))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.array([])
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        areas = np.append(areas, area)
        x, y, w, h = cv2.boundingRect(contours[i])
        rect = np.array([[x, y, w, h]])
        if i == 0:
            all_rects = rect
        else:
            all_rects = np.vstack([all_rects, rect])

    index = np.argsort(areas)
    areas = areas[index]
    all_rects = all_rects[index]
    # print(all_rects.shape)
    # print(areas.shape)
    # print(path)
    result = []

    selected_index = np.array([], dtype=np.int32)
    for i in range(len(all_rects) - 10, len(all_rects)):
        x = all_rects[i][0]
        y = all_rects[i][1]
        w = all_rects[i][2]
        h = all_rects[i][3]
        if y > 200 and y < 400:
            selected_img = img[y:y + h, x:x + w]
            result.append(selected_img)
            selected_index = np.append(selected_index, i)
    selected_rects = all_rects[selected_index]
    selected_areas = areas[selected_index]
    sorted_index = np.argsort(selected_rects[:, 0])
    return result, selected_rects, selected_areas, sorted_index


if __name__ == '__main__':
    root_path = "../../data"
    save_path = "../../data_3subimg"
    # result, selected_rects, selected_areas = boxCropForUltrasound("../../preprocessed_data/TrainSet/断裂/双侧/19.JPG")
    error_list = []
    count = 0
    for rt, dirs, files in os.walk(root_path):
        for file in files:
            if file[-4:].lower() in ['.png', '.bmp', '.JPG', '.Jpg', '.jpg']:
                path = os.path.join(rt, file)
                spath = os.path.join(save_path, rt[len(root_path) + 1:])
                if os.path.exists(os.path.join(spath, file)):
                    raise RuntimeError("file exists")

                result, selected_rects, selected_areas, img_index = boxCropForUltrasound(path)
                print(len(result))
                count += 1
                if len(result) != 3:
                    error_list.append(path)
                else:
                    if not os.path.isdir(spath):
                        os.makedirs(spath)

                    img1 = result[img_index[0]]
                    img2 = result[img_index[1]]
                    img3 = result[img_index[2]]
                    # print(selected_rects[img_index[0]],selected_rects[img_index[1]],selected_rects[img_index[2]])
                    for i in range(3):
                        sub_img = result[img_index[i]]
                        cv2.imwrite(os.path.join(spath, file[:-4] + f"-sub{i + 1}" + file[-4:]), sub_img)
                        # sub_img = Image.fromarray(sub_img)
                        # sub_img.save(os.path.join(spath, file[:-4]+f"-sub{i+1}"+file[-4:]))
    # print(error_list)
    print(len(error_list))
    print(count)
    with open("../../data_3subimg/uncroped_list.txt", "a+") as f:
        for line in error_list:
            f.write(line + "\n")