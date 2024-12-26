import os.path
from tkinter import filedialog, messagebox
import cv2
import torch
from consts import *
from imagedata_preprocessing import imagedata_preprocessing
from single_character_simpleCNN import transform, device, simpleCNN
from torchvision import transforms
from tqdm import tqdm
from sequence_handling import find_longest_sequence

model = simpleCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(os.path.join(PROJECT_PATH, r'saved_models', r'best_model_simpleCNN.pth')))
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
])

def test_full_image(image_path):
    image = imagedata_preprocessing(image_path)
    image = image.to(device)

    image = test_transform(image)
    image = image[None, :, :, :]

    segmented = [image[:, :, :, 5*i:5*(i+8)] for i in range(29)]
    predicted_characters = []
    model.eval()
    with torch.no_grad():
        # for i in tqdm(range(29)):
        for i in range(29):
            output = model(segmented[i])
            _, predicted = torch.max(output, 1)
            predicted_characters.append(CLASSES[predicted])

    result = find_longest_sequence(predicted_characters)
    return result

# test_full_image(os.path.join(DATAPATH, r'whole\all', r'0af622933aaf984998b09688a3e12401.2V8YV.jpeg'))
def performance_on_whole_test(path_to_data):
    total_count = 0
    correct_count = 0

    file_count = sum(len(files) for _, _, files in os.walk(path_to_data))
    with tqdm(total=file_count) as pbar:
        for root, dir, files in os.walk(path_to_data):
            for file in files:
                filename, ext = os.path.splitext(file)
                label = filename.split('.')[-1]
                label = label.to(device)
                predicted = ''.join(test_full_image(os.path.join(root, file)))
                # print(f"label: {label}, predicted: {test_full_image(os.path.join(root, file))}")
                if predicted == label:
                    correct_count += 1
                else:
                    print(f"Incorrect: {filename}, correct: {label}, predicted: {predicted}")
                total_count += 1
                pbar.update(1)

    print(f"Total images: {total_count}, correct images: {correct_count}")


performance_on_whole_test(os.path.join(DATAPATH, r'whole\test'))