from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from deeplabv3_cheryl.utils.CustomDataset import CustomDataset
from deeplabv3_cheryl.utils.metrics_functions import iou
import os
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Data Pre-processing"""

images_folder = '..\MoNuSeg\MoNuSeg 2018 Training Data\Images'
annotations_folder = '..\MoNuSeg\MoNuSeg 2018 Training Data\Annotations'

image_paths = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.tif')]
annotation_paths = [os.path.join(annotations_folder, f.replace('.tif', '.xml')) for f in os.listdir(images_folder) if f.endswith('.tif')]

train_image_paths, val_image_paths, train_annotation_paths, val_annotation_paths = train_test_split(
    image_paths, annotation_paths, test_size=0.15, random_state=42)

train_dataset = CustomDataset(
    image_paths=train_image_paths,
    annotation_paths=train_annotation_paths,
    mask_dir='data/train/mask'
)

val_dataset = CustomDataset(
    image_paths=val_image_paths,
    annotation_paths=val_annotation_paths,
    mask_dir='data/val/mask'
)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

"""Model Training"""
# Model, Loss function, Optimizer
model = deeplabv3_mobilenet_v3_large(pretrained=True)
model = model.to(device)

# Training and Validation Loops
num_epochs = 10
train_losses, val_losses = [], []
train_ious, val_ious = [], []

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Training Loop
    model.train()
    total_train_loss = 0.0
    total_train_iou = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)['out']

        masks = masks.expand(-1, 21, -1, -1)

        # Assuming masks are single-channel and Long type
        loss = criterion(outputs, masks.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()  # Accumulate the loss

        total_train_iou += iou(outputs.cpu(), masks.cpu()).item()

    train_losses.append(total_train_loss / len(train_loader))
    train_ious.append(total_train_iou / len(train_loader))

    # Validation Loop
    model.eval()
    total_val_loss, total_val_iou = 0.0, 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)['out']

            masks = masks.expand(-1, 21, -1, -1)

            total_val_loss += loss.item()  # Accumulate the loss

            total_val_iou += iou(val_outputs.cpu(), val_masks.cpu()).item()

    val_losses.append(total_val_loss / len(val_loader))
    val_ious.append(total_val_iou / len(val_loader))

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train IoU: {train_ious[-1]:.4f}, '
          f'Val Loss: {val_losses[-1]:.4f}, Val IoU: {val_ious[-1]:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_deeplabv3.pth')

# Plot the training and validation loss and IoU
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_ious, label='Train IoU')
plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU')
plt.title('IoU over Epochs')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()

plt.tight_layout()

plt.savefig('training_plot.jpg')

plt.show()
