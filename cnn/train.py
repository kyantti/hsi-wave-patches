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
from torch.utils.data import DataLoader, random_split
from cnn.util.helper_functions import plot_loss_curves
from cnn.data_setup import WaveletDataset

# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Setup directories
train_csv = "train_dataset.csv"
test_csv = "test_dataset.csv"

# Setup target device with id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create transforms
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Create a single instance of your dataset
wavelet_dataset = WaveletDataset(csv_file="dataset.csv", transform=transform)

img_labels = wavelet_dataset.img_labels

print("Image labels DataFrame:")
print(img_labels)
class_names = img_labels['class'].unique().tolist()
print(f"Class names: {class_names}")

# Decide on the split sizes
dataset_size = len(wavelet_dataset)
train_size = int(0.8 * dataset_size)  # 80% train
test_size = dataset_size - train_size  # 20% test

# Split the dataset
train_dataset, test_dataset = random_split(wavelet_dataset, [train_size, test_size])

# Create loaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup
weights = torchvision.models.ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights).to(device)

# Freeze all layers except the 2 head
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the number of features from the last layer
in_features = model.fc.in_features

# Replace the final classifier with a simple linear layer for 4 classes
model.fc = torch.nn.Linear(in_features=in_features, out_features=len(class_names), bias=True).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
summary(
    model,
    input_size=(
        BATCH_SIZE,
        3,
        224,
        224,
    ),  # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
    verbose=1,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# Start the timer
start_time = timer()

# Setup training and save the results
results = engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
    verbose=True,
    class_names=class_names,
)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

# Plot the training results
print("[INFO] Plotting training results...")
plot_loss_curves(results)
plt.savefig("out/figures/training_results.png", dpi=300, bbox_inches='tight')
plt.show()
print("[INFO] Training results plot saved as 'out/figures/training_results.png'")

# After training finishes
del model
del optimizer
del loss_fn
del train_dataloader
del test_dataloader

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
