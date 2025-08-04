"""
THIS CODE IMPLEMENTS MAIN PIPELINE AND MODEL TRAINING.
"""
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

import numpy as np
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm


DATA_DIR = "data"


class OpticSensorDataset(Dataset):
    def __init__(self, root, context_size=3, transform=None):
        self.img_dir = root
        self.context_size = context_size
        self.transform = transform or ToTensor()
        
        self.imgs = sorted(
            [f for f in os.listdir(root) if f.endswith('.png')],
            key=lambda name: int(re.search(r"frame_(\d+)", name).group(1))
        )
        self.img_paths = [os.path.join(root, fname) for fname in self.imgs]

        self.targets = [
            torch.tensor([float(x) for x in fname[:-4].split("_")[2:4]])
            for fname in self.imgs
        ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if index <= len(self) - self.context_size:
            indices = range(index, index + self.context_size)
        else:
            indices = [index] * self.context_size
        
        imgs = [self.transform(Image.open(self.img_paths[i]).convert("RGB")) for i in indices]
        img_concat = torch.cat(imgs, dim=0)

        target = self.targets[indices[-1]]
        return img_concat, target


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        
        self.block = nn.Sequential(
            # 1. Expansion
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # 2. Depthwise Conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # 3. Projection
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)
class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3*3):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.features = nn.Sequential(
            InvertedResidualBlock(32, 16, expansion=1, stride=1),
            InvertedResidualBlock(16, 24, expansion=6, stride=2),
            InvertedResidualBlock(24, 32, expansion=6, stride=2),
            InvertedResidualBlock(32, 64, expansion=6, stride=2),
            InvertedResidualBlock(64, 96, expansion=6, stride=1),
            InvertedResidualBlock(96, 96, expansion=1, stride=1),
            InvertedResidualBlock(96, 96, expansion=1, stride=1),
            InvertedResidualBlock(96, 96, expansion=1, stride=1),
            InvertedResidualBlock(96, 160, expansion=6, stride=2),
            InvertedResidualBlock(160, 320, expansion=6, stride=1),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 2)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

def train():
    # DataLoader
    datasetik = OpticSensorDataset(DATA_DIR, transform=ToTensor())
    dataloader = DataLoader(
        datasetik, 
        batch_size=64,
        num_workers=6,
        pin_memory=True
    )
    # Initializations
    mobilenet = MobileNetV2().to("cuda")
    # mobilenet.load_state_dict(torch.load("model_params.pth", weights_only=True)) - uncomment if finetuning
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=mobilenet.parameters(), lr=0.003)
    EPOCHS = 5
    # Training loop
    for epoch in range(EPOCHS):
        losses = []
        for imgs, targets in tqdm(dataloader):
            imgs = imgs.to("cuda")
            targets = targets.to("cuda")
            logits = mobilenet(imgs)
            outputs = torch.tanh(logits)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(np.array(losses).mean())
    
    torch.save(mobilenet.state_dict(), "model_params.pth")

if __name__ == "__main__":
    train()
    