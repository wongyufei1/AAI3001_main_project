import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from utils.CustomDataset import CustomDataset
from utils.fcn_model_wrapper import FCNModelWrapper
from utils.helper import plot_loss_iou
from utils.config import *

"""Data Pre-processing"""

train_images_folder = '../data/train/images'
train_annotations_folder = '../data/train/annotations'

val_images_folder = '../data/val/images'
val_annotations_folder = '../data/val/annotations'

train_image_paths = [os.path.join(train_images_folder, f) for f in os.listdir(train_images_folder) if
                     f.endswith('.tif')]
train_annotation_paths = [os.path.join(train_annotations_folder, f.replace('.tif', '.xml')) for f in
                          os.listdir(train_images_folder) if f.endswith('.tif')]

val_image_paths = [os.path.join(val_images_folder, f) for f in os.listdir(val_images_folder) if f.endswith('.tif')]
val_annotation_paths = [os.path.join(val_annotations_folder, f.replace('.tif', '.xml')) for f in
                        os.listdir(val_images_folder) if f.endswith('.tif')]

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
best_model = {"lrate": None, "wdecay": None, "transform": None, "epoch": None, "loss": None, "weights": None}

model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, progress=True)
model = model.to(DEVICE)

# Training and Validation Loops
learning_rates = [0.001, 0.0001]
weight_decays = [0.001, 0.0001]

for learning_rate in learning_rates:
    for weight_decay in weight_decays:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        model_wrapper = FCNModelWrapper(
            model=model,
            n_classes=N_CLASSES,
            device=DEVICE,
            optimizer=optimizer,
            epochs=EPOCHS,
            criterion=criterion
        )

        epoch, loss, iou, weights = model_wrapper.fit(train_loader, val_loader)

        if best_model["loss"] is None or best_model["loss"] > loss:
            best_model["lrate"] = learning_rate
            best_model["wdecay"] = weight_decay
            best_model["epoch"] = epoch
            best_model["loss"] = loss
            best_model["iou"] = iou
            best_model["weights"] = weights

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

plot_loss_iou(model_wrapper.train_losses, model_wrapper.val_losses, model_wrapper.train_ious, model_wrapper.val_ious,
              'loss_iou_plot.png')

print(f"Saving best model..."
      f"Learning Rate:{best_model['lrate']} Weight Decay:{best_model['wdecay']})"
      f"Epoch: {best_model['epoch']} Loss: {best_model['loss']}, IoU: {best_model['iou']}"
      )

# Save the trained model
torch.save(model.state_dict(), os.path.join(SAVE_PATH, "trained_fcn.pth"))
