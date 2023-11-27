import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from numpy import int32
from torch.utils.data import Dataset
from torchvision import tv_tensors


class MoNuSegDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_files = sorted(os.listdir(os.path.join(root, "Tissue Images")))
        self.mask_files = sorted(os.listdir(os.path.join(root, "Annotations")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Tissue Images", self.img_files[idx])
        mask_path = os.path.join(self.root, "Annotations", self.mask_files[idx])

        # load image standardized to RGB, generate instance masks, bboxes and labels
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        masks = tv_tensors.Mask(self.extract_masks(mask_path, (img_w, img_h)))
        bboxes = tv_tensors.BoundingBoxes(self.generate_bbox(masks), format="XYXY", canvas_size=img.size)
        labels = torch.ones((len(masks), ), dtype=torch.int64)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # img = tv_tensors.Image(img, dtype=torch.float32)

        # populate target with ground truths
        target = {
            "boxes": bboxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor(idx),
            "area": area,
            "iscrowd": torch.zeros((len(masks), ), dtype=torch.int64)
        }

        # transform img, masks and bboxes
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_files)

    def extract_masks(self, path, size):
        # print("Extracting masks...")
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

    def generate_bbox(self, masks):
        # print("Generating bounding boxes...")
        bboxes = []

        for mask in masks:
            coord = np.array(np.nonzero(mask))
            ymin, xmin = np.min(coord, axis=0)
            ymax, xmax = np.max(coord, axis=0)

            bboxes.append([xmin, ymin, xmax, ymax])

        return np.array(bboxes)
