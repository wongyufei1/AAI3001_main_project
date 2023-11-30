from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from deeplabv3_cheryl.utils.CustomDataset import CustomDataset
import os
import matplotlib.pyplot as plt
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from deeplabv3_cheryl.utils.model_wrapper import SemanticModelWrapper
from deeplabv3_cheryl.utils.config import N_CLASSES, DEVICE, EPOCHS




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
# Create a variables to store the overall best_model's weights, measure (accuracy) and epoch
best_hyperparameter = None
weights_chosen = None
best_measure = None
best_epochs = None

# Model, Loss function, Optimizer
model = deeplabv3_mobilenet_v3_large(weight = DeepLabV3_MobileNet_V3_Large_Weights)
model = model.to(DEVICE)
# model= fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, progress=True).to(device)

# Training and Validation Loops
learning_rates = [0.001, 0.0001]
weight_decays = [0.001, 0.0001]
train_losses, val_losses = [], []
train_ious, val_ious = [], []

for learning_rate in learning_rates:
    for weight_decay in weight_decays:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

        model_wrapper = SemanticModelWrapper(
            model = model,
            n_classes = N_CLASSES,
            device = DEVICE,
            optimizer = optimizer,
            epochs = EPOCHS,
            criterion = criterion
        )

        epoch, loss, weight = model_wrapper.fit(train_loader, val_loader)

        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Train IoU: {train_ious[-1]:.4f}, '
            f'Val Loss: {val_losses[-1]:.4f}, Val IoU: {val_ious[-1]:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_deeplabv3.pth')

"""Derrick's code"""
model= fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, progress=True).to(device)
model = model.to(DEVICE)








# Plot the training and validation loss and IoU
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_ious, label='Train IoU')
plt.plot(range(1, EPOCHS + 1), val_ious, label='Validation IoU')
plt.title('IoU over Epochs')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()

plt.tight_layout()

plt.savefig('training_plot.jpg')

plt.show()
