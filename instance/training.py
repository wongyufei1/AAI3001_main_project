# Evaluation code used in link below
# https://github.com/pytorch/vision/blob/main/references/detection/engine.py
import os

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from modules.config import *
from modules.dataset import MoNuSegDataset
from modules.model_wrapper import MRCNNModelWrapper
from modules.helper import *

if __name__ == "__main__":
    transforms = {
        "train_flip": T.Compose([
            T.RandomHorizontalFlip(),
        ]),
        "train_rotate": T.Compose([
            T.RandomRotation(10),
        ]),
        "train_contrast": T.Compose([
            T.RandomAutocontrast(),
        ])
    }

    dataset = {
        "train": MoNuSegDataset("./MoNuSeg/MoNuSeg 2018 Training Data", transforms=transforms["train_flip"]),
        "val": MoNuSegDataset("./MoNuSeg/MoNuSegTestData", transforms=None)
    }
    dataloader = {
        "train": DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(dataset["val"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    }

    best_model = {"lrate": None, "wdecay": None, "epoch": None, "loss": None, "weights": None}

    for lr in LRATES:
        for wd in WDECAYS:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

            optimizer = torch.optim.Adam(
                params=[p for p in model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=wd
            )

            model_wrapper = MRCNNModelWrapper(
                model=model,
                n_classes=N_CLASSES,
                device=DEVICE,
                optimizer=optimizer,
                epochs=EPOCHS
            )

            epoch, loss, weights = model_wrapper.fit(train_loader=dataloader["train"], val_loader=dataloader["val"])

            if best_model["loss"] is None or best_model["loss"] > loss:
                best_model["lrate"] = lr
                best_model["wdecay"] = wd
                best_model["epoch"] = epoch
                best_model["loss"] = loss
                best_model["weights"] = weights

    print(f"Best model: {best_model}")
    torch.save(best_model["weights"], "model.pt")
