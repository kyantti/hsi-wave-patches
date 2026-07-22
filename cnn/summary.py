"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
from cnn import engine
import torchvision
import gc
from torchinfo import summary
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from cnn.util.helper_functions import plot_loss_curves

# Setup hyperparameters
experiment_num = 2
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Setup target device with id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dir = "data/processed/train"
test_dir = "data/processed/test"

# 2. Keep the separate transforms for training and testing
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=train_dir, transform=train_transform)
test_dataset = ImageFolder(root=test_dir, transform=test_transform)

class_names = train_dataset.classes
print(f"Class names found: {class_names}")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

weights = torchvision.models.DenseNet121_Weights.DEFAULT
model = torchvision.models.densenet121(weights=weights).to(device)

in_features = model.classifier.in_features

# Replace the final classifier with the custom head
model.classifier = torch.nn.Sequential( # type: ignore
    torch.nn.Linear(in_features, 128),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(128),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(64, len(class_names))
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Get image shape from the dataset
img, _ = train_dataset[0]
C, H, W = img.shape

summary(
    model,
    input_size=(
        BATCH_SIZE,
        C,
        H,
        W,
    ),
    verbose=1,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# After training finishes
del model
del optimizer
del loss_fn
del train_dataloader
del test_dataloader

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()