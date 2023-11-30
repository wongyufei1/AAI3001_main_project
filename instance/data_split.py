import os
import shutil

from torch.utils.data import random_split

from utils.config import ORIG_DATA_PATH, NEW_DATA_PATH

if __name__ == "__main__":
    # brain and lung images are not included in train/val data
    train_val_files = {
        "breast": ["TCGA-A7-A13E-01Z-00-DX1", "TCGA-A7-A13F-01Z-00-DX1", "TCGA-AR-A1AK-01Z-00-DX1",
                   "TCGA-AR-A1AS-01Z-00-DX1", "TCGA-E2-A1B5-01Z-00-DX1", "TCGA-E2-A14V-01Z-00-DX1"],
        "kidney": ["TCGA-B0-5711-01Z-00-DX1", "TCGA-HE-7128-01Z-00-DX1", "TCGA-HE-7129-01Z-00-DX1",
                   "TCGA-HE-7130-01Z-00-DX1", "TCGA-B0-5710-01Z-00-DX1", "TCGA-B0-5698-01Z-00-DX1"],
        "liver": ["TCGA-18-5592-01Z-00-DX1", "TCGA-38-6178-01Z-00-DX1", "TCGA-49-4488-01Z-00-DX1",
                  "TCGA-50-5931-01Z-00-DX1", "TCGA-21-5784-01Z-00-DX1", "TCGA-21-5786-01Z-00-DX1"],
        "prostate": ["TCGA-G9-6336-01Z-00-DX1", "TCGA-G9-6348-01Z-00-DX1", "TCGA-G9-6356-01Z-00-DX1",
                     "TCGA-G9-6363-01Z-00-DX1", "TCGA-CH-5767-01Z-00-DX1", "TCGA-G9-6362-01Z-00-DX1"],
        "bladder": ["TCGA-DK-A2I6-01A-01-TS1", "TCGA-G2-A2EK-01A-02-TSB"],
        "colon": ["TCGA-AY-A8YK-01A-01-TS1", "TCGA-NH-A8F7-01A-01-TS1"],
        "stomach": ["TCGA-KB-A93J-01A-01-TS1", "TCGA-RD-A8N9-01A-01-TS1"]
    }

    data_files = {
        "train": [],
        "val": []
    }

    for organ, files in train_val_files.items():
        train, val = random_split(files, (0.7, 0.3))
        data_files["train"].extend([f for f in train])
        data_files["val"].extend([f for f in val])

    old_img_dir = os.path.join(ORIG_DATA_PATH, "Images")
    old_anno_dir = os.path.join(ORIG_DATA_PATH, "Annotations")

    for split, files in data_files.items():
        img_dir = os.path.join(NEW_DATA_PATH, split, "images")
        anno_dir = os.path.join(NEW_DATA_PATH, split, "annotations")

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)

        for file in files:
            old_img_path = os.path.join(old_img_dir, file + ".tif")
            old_anno_path = os.path.join(old_anno_dir, file + ".xml")
            img_path = os.path.join(img_dir, file + ".tif")
            anno_path = os.path.join(anno_dir, file + ".xml")

            shutil.copyfile(old_img_path, img_path)
            shutil.copyfile(old_anno_path, anno_path)



