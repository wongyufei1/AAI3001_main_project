import os

import torchvision
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from utils.config import *
from utils.dataset import MoNuSegDataset
from utils.model_wrapper import MRCNNModelWrapper
from utils.helper import *

if __name__ == "__main__":
    print("Defining transforms...")
    train_transforms = {
        "default": T.Compose([
            # T.Resize(256, antialias=True),
            T.ToDtype(dtype={tv_tensors.Image: torch.float32, "others": None}, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "contrast": T.Compose([
            # T.Resize(256, antialias=True),
            T.RandomAutocontrast(),
            T.ToDtype(dtype={tv_tensors.Image: torch.float32, "others": None}, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    val_transforms = T.Compose([
        # T.Resize(256, antialias=True),
        T.ToDtype(dtype={tv_tensors.Image: torch.float32, "others": None}, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    """
        Train and select model based on transform, learning rate, weight decay
    """
    best_model = {"lrate": None, "wdecay": None, "transform": None, "epoch": None, "loss": None, "weights": None}

    for t_name, transform in train_transforms.items():
        dataset = {
            "train": MoNuSegDataset("../data/train", transforms=transform),
            "val": MoNuSegDataset("../data/val", transforms=val_transforms)
        }

        dataloader = {
            "train": DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn),
            "val": DataLoader(dataset["val"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        }

        for lr in LRATES:
            for wd in WDECAYS:
                print(f"Training model...(Transform:{t_name} Learning Rate:{lr} Weight Decay:{wd})")

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
                    epochs=EPOCHS,
                )

                epoch, loss, weights = model_wrapper.fit(train_loader=dataloader["train"], val_loader=dataloader["val"])

                if best_model["loss"] is None or best_model["loss"] > loss:
                    best_model["lrate"] = lr
                    best_model["wdecay"] = wd
                    best_model["transform"] = t_name
                    best_model["epoch"] = epoch
                    best_model["loss"] = loss
                    best_model["weights"] = weights

    """
        Plot and save train and validation loss curves, and save model weights
    """
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # plot and save train and validation loss curves
    plot_loss(model_wrapper.train_losses, model_wrapper.val_losses, os.path.join(SAVE_PATH, "loss.png"))

    print(f"Saving best model...(Transform:{best_model['transform']} "
          f"Learning Rate:{best_model['lrate']} Weight Decay:{best_model['wdecay']})")
    torch.save(best_model["weights"], os.path.join(SAVE_PATH, "model.pt"))
