import cv2
from glob import glob
import os
from tqdm import tqdm
from PIL import Image

def boxCropForUltrasound(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    answer = img.copy()

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        if h > 130 and w > 50:
            answer = img[y:y+h, x:x+w]
        else:
            continue
    answer[:100, 1000:] = 0
    return answer


if __name__ == '__main__':

    root_path = "../../zzszqm"
    save_path = "../../new_data"

    for rt, dirs, files in os.walk(root_path):
        for file in tqdm(files):
            if file[-4:] in [r'.png', r'.bmp', r'.JPG', r'.jpg']:
                path = os.path.join(rt, file)
                spath = os.path.join(save_path, rt[len(root_path) + 1:])
                if os.path.isfile(os.path.join(spath, file)):
                    continue

                img = boxCropForUltrasound(path)
                if not os.path.isdir(spath):
                    os.makedirs(spath)

                img = Image.fromarray(img)
                img.save(os.path.join(spath, file))