import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_dir, annotation_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.annotation_paths = annotation_paths
        self.transform = transform or transforms.ToTensor()

        if annotation_paths is not None:
            self.masks = self._create_masks()
        else:
            self.masks = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.masks is not None:
            mask = Image.fromarray(self.masks[idx])
        else:
            mask = None

        img = self.transform(img)
        if mask is not None:
            mask = self.transform(mask)

        return img, mask

    def _create_masks(self):
        masks = []

        for img_path, annotation_path in zip(self.image_paths, self.annotation_paths):
            mask = self._xml_to_mask(annotation_path, Image.open(img_path).size)
            masks.append(mask)

        return masks

    def _xml_to_mask(self, xml_file, img_shape):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        mask = np.zeros(img_shape[:2], dtype=np.uint8)  # Assuming img_shape is in (H, W) format

        for region in root.iter('Region'):
            polygon = []
            for vertex in region.iter('Vertex'):
                x = int(float(vertex.get('X')))
                y = int(float(vertex.get('Y')))
                polygon.append((x, y))

            np_polygon = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(mask, np_polygon, 255)  # Fill polygon with 255

        return mask

