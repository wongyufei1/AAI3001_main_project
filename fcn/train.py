import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from sklearn.model_selection import train_test_split
from modules.dataset import MoNuSegDataset
from modules.metrics import iou
import os
import json
import matplotlib.pyplot as plt

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA
image_dir = '..\MoNuSeg\MoNuSeg 2018 Training Data\Images'
annotation_dir = '..\MoNuSeg\MoNuSeg 2018 Training Data\Annotations'

image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')]
annotation_paths = [os.path.join(annotation_dir, f.replace('.tif', '.xml')) for f in os.listdir(image_dir) if f.endswith('.tif')]

test_image_dir = '..\MoNuSeg\MoNuSegTestData\Images'
test_annotation_dir = '..\MoNuSeg\MoNuSegTestData\Annotations'
test_image_paths = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith('.tif')]
test_annotation_paths = [os.path.join(test_annotation_dir, f.replace('.tif', '.xml')) for f in os.listdir(test_annotation_dir) if f.endswith('.tif')]

train_image_paths, val_image_paths, train_annotation_paths, val_annotation_paths = train_test_split(
    image_paths, annotation_paths, test_size=0.2, random_state=42)



train_dataset = MoNuSegDataset(
    image_paths=train_image_paths,
    annotation_paths=train_annotation_paths,
    mask_dir='masks/train'
)

val_dataset = MoNuSegDataset(
    image_paths=val_image_paths,
    annotation_paths=val_annotation_paths,
    mask_dir='masks/val'
)

test_dataset = MoNuSegDataset(
    image_paths=test_image_paths,
    annotation_paths=test_annotation_paths,
    mask_dir='masks/test'
)
with open('train_images.json', 'w') as f:
    json.dump(train_image_paths, f)
with open('val_images.json', 'w') as f:
    json.dump(val_image_paths, f)
with open('test_images.json', 'w') as f:
    json.dump(test_image_paths, f)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# print(len(train_loader))
# Model, Loss function, Optimizer
model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, progress=True).to(device)
model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Validation Loops
num_epochs = 2
train_losses, val_losses = [], []
train_ious, val_ious = [], []

for epoch in range(num_epochs):
    # Training Loop
    model.train()
    total_train_loss = 0.0
    total_train_iou = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        total_train_loss += loss.item()

        preds = torch.sigmoid(outputs) > 0.5
        masks_binary = masks > 0.5  # Convert masks to binary
        total_train_iou += iou(preds, masks_binary).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(total_train_loss / len(train_loader))
    train_ious.append(total_train_iou / len(train_loader))

    # Validation Loop
    model.eval()
    total_val_loss, total_val_iou = 0.0, 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)['out']
            val_loss = criterion(val_outputs, val_masks)
            total_val_loss += val_loss.item()

            val_preds = torch.sigmoid(val_outputs) > 0.5
            total_val_iou += iou(val_preds, val_masks).item()

    val_losses.append(total_val_loss / len(val_loader))
    val_ious.append(total_val_iou / len(val_loader))

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train IoU: {train_ious[-1]:.4f}, '
          f'Val Loss: {val_losses[-1]:.4f}, Val IoU: {val_ious[-1]:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_fcn.pth')

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
plt.savefig('training_plots.png')
plt.show()
