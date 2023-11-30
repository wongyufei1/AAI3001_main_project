import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt


def plot_loss_iou(train_losses, val_losses, train_ious, val_ious, path):
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # plot train and validation losses
    sns.lineplot(x=range(len(train_losses)),
                 y=train_losses,
                 ax=axes[0], label="Train loss")
    sns.lineplot(x=range(len(val_losses)),
                 y=val_losses,
                 ax=axes[0], label="Valid loss")

    axes[0].set_title("Best Model's Average Losses Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Average Loss")

    train_ious = np.array(train_ious)
    val_ious = np.array(val_ious)

    # plot train and validation IoU
    sns.lineplot(x=range(len(train_ious)),
                 y=train_ious,
                 ax=axes[1], label="Train IoU")
    sns.lineplot(x=range(len(val_ious)),
                 y=val_ious,
                 ax=axes[1], label="Valid IoU")

    axes[1].set_title("Best Model's Average IoU Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Average IoU")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
