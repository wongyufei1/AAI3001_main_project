import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET

from numpy import int32


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

        if area >= 1:
            bboxes.append(bbox)
            areas.append(area)
            save_masks.append(mask)

    return np.array(bboxes), np.array(areas), np.array(save_masks)


def collate_fn(batch):
    imgs = torch.stack([img for img, target in batch])
    targets = [target for img, target in batch]
    return [imgs, targets]
