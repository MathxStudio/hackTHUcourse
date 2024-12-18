import torch
import cv2
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms


from PIL import Image
import numpy as np

from consts import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
loss_function = torch.nn.CrossEntropyLoss()
transform = transforms.Compose([
    transforms.ToTensor(),
])  # Ensure dtype of image is correct

class SingleCharacterDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.transform = transform
        self.data_folder = data_folder

        self.image_with_labels = [] # len x 2

        for root, dir, files in os.walk(data_folder):
            relpath = os.path.relpath(root, data_folder)
            for filename in files:
                name, ext = os.path.splitext(filename)
                name = name.split('.')
                if ext == '.jpeg' and len(name) == 3 and len(name[0]) == 32:
                    self.image_with_labels.append((filename, relpath))

    def __getitem__(self, index):
        filename, label = self.image_with_labels[index]
        image_path = os.path.join(self.data_folder, label, filename)
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)

        label_index = [CLASSES_TO_ID[_label] for _label in label]
        label_tensor = torch.LongTensor(label_index).squeeze()
        return image, label_tensor

    def __len__(self):
        return len(self.image_with_labels)

    # def returnsample(self):
    #     return self.image_with_labels

# hello = SingleCharacterDataset(TRAIN_SAMPLES_PATH)
# print(hello.returnsample()[0])
# print(hello[0], hello[1])

single_character_train_dataset = SingleCharacterDataset(TRAIN_SAMPLES_PATH, transform=transform)
single_character_test_dataset = SingleCharacterDataset(TEST_SAMPLES_PATH, transform=transform)
single_character_train_dataloader = DataLoader(dataset=single_character_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
single_character_test_dataloader = DataLoader(dataset=single_character_test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(  # 3 x 40 x 40
            nn.Conv2d(3, 32, kernel_size=(5,5)), # 32 x 36 x 36
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32 x 18 x 18
            nn.Conv2d(32, 64, kernel_size=(3,3)), # 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 64 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=(3,3)), # 128 x 6 x 6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 128 x 3 x 3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

net = AlexNet(num_classes).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

history = {'Test Loss': [], 'Test Accuracy': [], 'Train Loss': [], 'Train Accuracy': []}

def train_model():
    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        net.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        for image, label in tqdm(single_character_train_dataloader, desc="Training", total=len(single_character_train_dataloader)):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = net(image)

            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train_correct += (predicted == label).sum().item()
            total_train_samples += label.size(0)

        avg_train_loss = total_train_loss / len(single_character_train_dataloader)
        avg_train_accuracy = total_train_correct / total_train_samples

        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
        history['Train Loss'].append(avg_train_loss)
        history['Train Accuracy'].append(avg_train_accuracy)

        net.eval()
        total_test_loss = 0
        total_test_correct = 0
        total_test_samples = 0

        for image, label in tqdm(single_character_test_dataloader, desc="Testing", total=len(single_character_test_dataloader)):
            image, label = image.to(device), label.to(device)

            with torch.no_grad():
                outputs = net(image)

            loss = loss_function(outputs, label)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test_correct += (predicted == label).sum().item()
            total_test_samples += label.size(0)

        avg_test_loss = total_test_loss / len(single_character_test_dataloader)
        avg_test_accuracy = total_test_correct / total_test_samples

        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
        history['Test Loss'].append(avg_test_loss)
        history['Test Accuracy'].append(avg_test_accuracy)

        if avg_test_accuracy > best_acc:
            best_acc = avg_test_accuracy
            torch.save(net.state_dict(), 'best_model.pth')


        plt.figure(figsize=(8, 6))
        plt.plot(history['Train Loss'], label='Train Loss')
        plt.plot(history['Test Loss'], label='Test Loss')
        plt.plot(history['Train Accuracy'], label='Train Accuracy')
        plt.plot(history['Test Accuracy'], label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.title(f'Epoch {epoch + 1} Loss/Accuracy Trend')
        plt.show()


# train_model()


def example():
    sample_image_path = os.path.join(DATAPATH, r'segmented_afterpre\train\4\00bebc089ae21814743963b8d0ec3422.Q9X4J.124.jpeg')

    image = cv2.imread(sample_image_path)
    print(image)
    print(image.size)
    print(image.shape)
    newimg = Image.open(sample_image_path).convert('L')
    print(np.array(newimg).shape)
    # cv2.imshow('sample image', image)
    # newimg.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

