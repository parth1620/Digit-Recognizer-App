import torch 
from torch import nn

class DigitModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))

        )

        self.linear_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, 10)
        )

    def forward(self, images, labels = None):

        x = self.cnn_block(images)
        logits = self.linear_block(x)

        if labels != None:
            return logits, nn.CrossEntropyLoss()(logits, labels)

        return logits
