from tkinter import filedialog, messagebox
import cv2
import torch
from consts import *
from imagedata_preprocessing import imagedata_preprocessing
from single_character import transform, device, AlexNet
from torchvision import transforms



model = AlexNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(os.path.join(PROJECT_PATH, r'best_model.pth')))
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
])

def test_model():
    # test_file_path = filedialog.askopenfilename(title="Select an image for testing:")
    test_file_path = r'G:\image_recognition\dataset\segmented\all\3\0a4e71ea3cf377c073fd710519ca7e15.E2VX3.140.jpeg'
    image = imagedata_preprocessing(test_file_path)
    image = test_transform(image)
    image = image[None, :, :, :]



    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_character = CLASSES[predicted]
        messagebox.showinfo("Prediction result", f"Predicted character: {predicted_character}")

# test_model()
