import cv2
import os
import shutil
import numpy as np

from tqdm import tqdm
from consts import *


def imagedata_preprocessing(path_to_image):
    image = cv2.imread(path_to_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    edges = cv2.Canny(opening, 50, 150)

    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    return dilated

def example():
    image_path = os.path.join(DATAPATH, r'segmented\all\C\0a8761bdc49e535f315a25c14727e80d.PW9C.144.jpeg')
    image = cv2.imread(image_path)
    processed_image = imagedata_preprocessing(image_path)

    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_all(path_to_images, output_path):
    file_count = sum(len(files) for _, _, files in os.walk(path_to_images))
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    with tqdm(total=file_count) as pbar:
        for root, dirs, files in os.walk(path_to_images):
            for name in files:
                filename, ext = os.path.splitext(name)
                if ext.endswith(('.png', 'jpg', 'jpeg')):
                    relpath = os.path.relpath(root, path_to_images)
                    curpath = os.path.join(root, name)
                    processed_image = imagedata_preprocessing(curpath)
                    outpath = os.path.join(output_path, relpath, name)
                    if not os.path.exists(os.path.join(output_path, relpath)):
                        os.makedirs(os.path.join(output_path, relpath))

                    cv2.imwrite(outpath, processed_image)
                    # print(f"Processed {curpath} -> {outpath}")
                pbar.update(1)

# process_all(SEGMENTED_aftersampling, SEGMENTED_afterpre)
example()