import os
import torchvision

from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from utils.config import *
from utils.dataset import MoNuSegDataset
from utils.helper import collate_fn, draw_predictions
from utils.model_wrapper import MRCNNModelWrapper

if __name__ == "__main__":
    print("Defining transforms...")
    transforms = T.Compose([
        T.Resize(256, antialias=True),
        T.ToDtype(dtype={tv_tensors.Image: torch.float32, "others": None}, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Evaluating model inference time...")
    dataset = MoNuSegDataset("../data/test_combined", transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    model_wrapper = MRCNNModelWrapper(
        model=model,
        n_classes=N_CLASSES,
        device=DEVICE,
        weights=torch.load(os.path.join(SAVE_PATH, "best_model.pt"), map_location=DEVICE)
    )

    imgs, predictions, infer_time = model_wrapper.predict_batches(dataloader=dataloader, timer=True)

    print(f"\nEvaluation results for... \nData Size:{len(dataset)}\nBatch Size:{BATCH_SIZE}\nDevice: {DEVICE}")
    print(f"Total inference time: {infer_time}")
    print(f"Average inference time per batch: {infer_time / BATCH_SIZE}")

    """
        Draw and save predictions
    """
    if not os.path.exists(os.path.join(SAVE_PATH, "predictions")):
        os.makedirs(os.path.join(SAVE_PATH, "predictions"))

    # draw and save predictions
    for idx, img in enumerate(imgs):
        save_img = draw_predictions(img, predictions[idx])

        # save predicted images
        save_img.save(os.path.join(SAVE_PATH, "predictions", "test_" + str(idx)) + ".png")
