import torchvision
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from utils.config import *
from utils.dataset import MoNuSegDataset
from utils.helper import collate_fn
from utils.model_wrapper import MRCNNModelWrapper

if __name__ == "__main__":
    print("Defining transforms...")
    transforms = T.Compose([
        # T.Resize(256, antialias=True),
        T.ToDtype(dtype={tv_tensors.Image: torch.float32, "others": None}, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_splits = ["test_in_train", "test_not_in_train", "test_combined"]

    for split in test_splits:
        print(f"Evaluating model...(Set:{split} Metric:mAP)")
        dataset = MoNuSegDataset(f"../data/{split}", transforms=transforms)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        model_wrapper = MRCNNModelWrapper(
            model=model,
            n_classes=N_CLASSES,
            device=DEVICE,
            weights=torch.load(os.path.join(SAVE_PATH, "best_model.pt"), map_location=DEVICE)
        )

        model_wrapper.evaluate(dataloader=dataloader)
        print()
