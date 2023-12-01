import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt

from numpy import int32
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def extract_masks(path, size):
    img_masks = []

    tree = ET.parse(path)
    root = tree.getroot()

    for instance in root.findall("Annotation/Regions/Region"):
        mask = np.zeros((size[1], size[0]))
        points = np.array(
            [[float(v.get("X")), float(v.get("Y"))] for v in instance.findall("Vertices/Vertex")],
            dtype=int32
        )
        cv2.fillPoly(mask, [points], 1)

        img_masks.append(mask)

        # cv2.imshow("mask", mask)
        # cv2.waitKey()
    # cv2.destroyAllWindows()

    return np.array(img_masks)


def generate_bbox(masks):
    bboxes = []
    areas = []
    save_masks = []

    for mask in masks:
        coord = np.array(np.nonzero(mask))
        ymin, xmin = np.min(coord, axis=0)
        ymax, xmax = np.max(coord, axis=0)
        bbox = [xmin, ymin, xmax, ymax]
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

        if xmin < xmax and ymin < ymax:
            bboxes.append(bbox)
            areas.append(area)
            save_masks.append(mask)

    return np.array(bboxes), np.array(areas), np.stack(save_masks)


def collate_fn(batch):
    imgs = torch.stack([img for img, target in batch])
    targets = [target for img, target in batch]
    return [imgs, targets]


def plot_loss(train_losses, val_losses, path):
    figure, ax = plt.subplots(1, 1, figsize=(10, 7))

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # plot train and validation losses
    sns.lineplot(x=range(len(train_losses)),
                 y=train_losses,
                 ax=ax, label="Train loss")
    sns.lineplot(x=range(len(val_losses)),
                 y=val_losses,
                 ax=ax, label="Valid loss")

    ax.set_title("Best Model's Average Losses Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")

    plt.savefig(path, bbox_inches="tight")
    plt.show()


def draw_predictions(img, prediction):
    # transform to inverse the normalisation on the images
    inv_norm = T.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225])

    # scale up image
    img = inv_norm(img).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

    labels = [f"Score: {s:.3f}" for s in prediction["scores"]]

    # draw bboxes
    bboxes = prediction["boxes"].long()
    output_img = draw_bounding_boxes(img, bboxes, labels, colors="red")

    # draw masks
    masks = (prediction["masks"] > 0.7).squeeze(1)
    output_img = draw_segmentation_masks(output_img, masks, alpha=0.5, colors="blue")

    return Image.fromarray(output_img.permute(1, 2, 0).numpy())
