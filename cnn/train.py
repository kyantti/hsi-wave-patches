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

torch.manual_seed(42)
torch.cuda.manual_seed(42)

start_time = timer()

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
    scheduler=scheduler,
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

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

# Save results
print("[INFO] Plotting training results...")
plot_loss_curves(results)
training_results_filename = f"out/figures/experiment_{experiment_num}_training_results.png"
plt.savefig(training_results_filename, dpi=300, bbox_inches='tight')
plt.show()
print(f"[INFO] Training results plot saved as '{training_results_filename}'")

if "classification_report" in results:
    report = results["classification_report"][0]
    df_report = pd.DataFrame(report).transpose()
    report_filename = f"out/tables/experiment_{experiment_num}_classification_report.csv"
    df_report.to_csv(report_filename, index=True)
    print(f"📝 Classification report saved to '{report_filename}'")

# Save and Plot Confusion Matrix
if "confusion_matrix" in results:
    cm = results["confusion_matrix"][0]
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
                
    # Save to CSV
    cm_filename_csv = f"out/tables/experiment_{experiment_num}_confusion_matrix.csv"
    df_cm.to_csv(cm_filename_csv, index=True)
    print(f"📋 Confusion matrix saved to '{cm_filename_csv}'")
                
    # Plot and save as PNG
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    cm_filename_png = f"out/figures/experiment_{experiment_num}_confusion_matrix.png"
    plt.savefig(cm_filename_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"🖼️ Confusion matrix plot saved to '{cm_filename_png}'")


