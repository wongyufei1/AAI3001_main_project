import torchvision.transforms.v2 as T

from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from instance.utils.helper import *


class MoNuSegDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_files = sorted(os.listdir(os.path.join(root, "Images")))[:1]
        self.mask_files = sorted(os.listdir(os.path.join(root, "Annotations")))[:1]
        self.norm = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.img_files[idx])
        mask_path = os.path.join(self.root, "Annotations", self.mask_files[idx])

        # load image standardized to RGB, generate instance masks, bboxes and labels
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        masks = tv_tensors.Mask(extract_masks(mask_path, (img_w, img_h)))
        bboxes, areas, masks = generate_bbox(masks)
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=img.size)
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

        # apply normalisation to image only
        img = self.norm(img)

        # transform img, masks and bboxes
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_files)

