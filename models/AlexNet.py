import torch
import torch.nn as nn

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

