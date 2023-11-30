import os

from torch.utils.data import Dataset
from torchvision import tv_tensors

from instance.utils.helper import *


class MoNuSegDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_files = sorted(os.listdir(os.path.join(root, "images")))
        self.mask_files = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.img_files[idx])
        mask_path = os.path.join(self.root, "annotations", self.mask_files[idx])

        # load image standardized to RGB, generate instance masks, bboxes and labels
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        img = tv_tensors.Image(img)
        masks = tv_tensors.Mask(extract_masks(mask_path, (img_w, img_h)))
        bboxes, areas, masks = generate_bbox(masks)
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(img_h, img_w))
        masks = tv_tensors.Mask(masks, dtype=torch.uint8)
        labels = torch.ones((len(masks), ), dtype=torch.int64)

        # populate target with ground truths
        target = {
            "boxes": bboxes,
            "labels": labels,
            "masks": masks,
            "image_id": idx,
            "area": torch.as_tensor(areas),
            "iscrowd": torch.zeros((len(masks), ), dtype=torch.int64)
        }

        # transform img, masks and bboxes
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_files)

