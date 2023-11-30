import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from utils.config import *
from utils.dataset import MoNuSegDataset
from utils.helper import collate_fn
from utils.model_wrapper import MRCNNModelWrapper

if __name__ == "__main__":
    dataset = MoNuSegDataset("./MoNuSeg/MoNuSegTestData", transforms=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model_wrapper = MRCNNModelWrapper(
        model=model,
        n_classes=N_CLASSES,
        device=DEVICE,
        weights=torch.load("model.pt", map_location=DEVICE)
    )

    model_wrapper.evaluate(dataloader=dataloader)

    norm = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open("./MoNuSeg/MoNuSegTestData/Images/TCGA-2Z-A9J9-01A-01-TS1.tif").convert("RGB")
    img = norm(img)
    predictions = model_wrapper.predict(img)

    inv_norm = T.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225])
    img = (inv_norm(img) * 255.0).to(torch.uint8)

    labels = [f"Nucleus: {score:.3f}" for label, score in zip(predictions[0]["labels"],
                                                              predictions[0]["scores"])]
    bboxes = predictions[0]["boxes"].long()
    output_img = draw_bounding_boxes(img, bboxes, labels, colors="red")

    masks = (predictions[0]["masks"] > 0.7).squeeze(1)
    output_img = draw_segmentation_masks(output_img, masks, alpha=0.5, colors="blue")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_img.permute(1, 2, 0))
    plt.show()
