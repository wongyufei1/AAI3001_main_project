import os
import shutil

from torch.utils.data import random_split

from config import ORIG_TRAIN_PATH, NEW_DATA_PATH, TRAIN_VAL_FILES, TEST_FILES, ORIG_TEST_PATH

if __name__ == "__main__":
    data_files = {
        "train": [],
        "val": []
    }

    """
        Split train and validation data
    """
    for organ, files in TRAIN_VAL_FILES.items():
        train, val = random_split(files, (0.7, 0.3))
        data_files["train"].extend([f for f in train])
        data_files["val"].extend([f for f in val])

    """
        Save train and validation data
    """
    old_img_dir = os.path.join(ORIG_TRAIN_PATH, "Images")
    old_anno_dir = os.path.join(ORIG_TRAIN_PATH, "Annotations")

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

    """
        Split test data into organs in train/val and not in train/val
    """
    # to check for disjoint train/val/test split
    val_files = os.listdir(os.path.join(NEW_DATA_PATH, "val", "images"))
    train_files = os.listdir(os.path.join(NEW_DATA_PATH, "train", "images"))

    old_img_dir = os.path.join(ORIG_TEST_PATH, "Images")
    old_anno_dir = os.path.join(ORIG_TEST_PATH, "Annotations")

    test_splits = ["test_in_train", "test_not_in_train", "test_combined"]

    # create the directories for 3 splits of test data
    for split in test_splits:
        split_img_dir = os.path.join(NEW_DATA_PATH, split, "images")
        split_anno_dir = os.path.join(NEW_DATA_PATH, split, "annotations")
        if not os.path.exists(split_img_dir):
            os.makedirs(split_img_dir)

        if not os.path.exists(split_anno_dir):
            os.makedirs(split_anno_dir)

    # create 3 splits of test data
    for organ, files in TEST_FILES.items():
        for file in files:
            # check for train/val/test data disjoint
            if (file + ".tif") in train_files or (file + ".tif") in val_files:
                print(f"The following file is not disjoint with train/val data: {file}.tif")
                continue;

            old_img_path = os.path.join(old_img_dir, file + ".tif")
            old_anno_path = os.path.join(old_anno_dir, file + ".xml")

            # always add into the combined test data
            img_path = os.path.join(NEW_DATA_PATH, "test_combined", "images", file + ".tif")
            anno_path = os.path.join(NEW_DATA_PATH, "test_combined", "annotations", file + ".xml")
            shutil.copyfile(old_img_path, img_path)
            shutil.copyfile(old_anno_path, anno_path)

            # add brain and lung organ data into test that are not in train/val
            if organ == "brain" or organ == "lung":
                img_path = os.path.join(NEW_DATA_PATH, "test_not_in_train", "images", file + ".tif")
                anno_path = os.path.join(NEW_DATA_PATH, "test_not_in_train", "annotations", file + ".xml")
                shutil.copyfile(old_img_path, img_path)
                shutil.copyfile(old_anno_path, anno_path)
            else:
                img_path = os.path.join(NEW_DATA_PATH, "test_in_train", "images", file + ".tif")
                anno_path = os.path.join(NEW_DATA_PATH, "test_in_train", "annotations", file + ".xml")
                shutil.copyfile(old_img_path, img_path)
                shutil.copyfile(old_anno_path, anno_path)

