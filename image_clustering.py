import PIL.ImageShow
import torch
import cv2
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms


from PIL import Image
import numpy as np
import pandas as pd

from consts import *
# from imagedata_preprocessing import imagedata_preprocessing

# initial_centers = [[20, 18], [20, 54], [20, 90], [20, 126], [20, 162]]
WB_THRESHOLD = 0.5 * 255

def vertical_projection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    binary = torch.tensor(binary, dtype=torch.float32)

    vertical_projection = torch.sum(binary, dim=0)
    normalized_projection = (vertical_projection - vertical_projection.min()) / (
                vertical_projection.max() - vertical_projection.min())
    normalized_np = normalized_projection.numpy()
    df = pd.DataFrame(normalized_np)
    df.to_csv("test_csv", index=False)

    peak_threshold = 0.3  # This value may need adjustment based on image characteristics

    peaks = (normalized_projection > peak_threshold).float()
    transitions = torch.diff(peaks)
    character_starts = torch.sum(transitions == 1).item()

    # Estimate the number of characters
    estimated_characters = character_starts
    print(estimated_characters)  # 40 x 180

def CCA(image_path):
    processed_image = imagedata_preprocessing(image_path)
    components = cv2.connectedComponents(processed_image)
    return components


def example():
    example_image_path = os.path.join(UNSEGMENTED_PATH, r'all', r'00a475e6ceb1fce7d622fa6210abbeb8.K8V2M.jpeg')
    vertical_projection(example_image_path)

example()