import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T

from dataset import MoNuSegDataset


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    epochs = 5

    transforms = {
        "train": T.Compose([
            T.Resize(256),
            # T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    dataset = MoNuSegDataset("./MoNuSeg/MoNuSeg 2018 Training Data", transforms=transforms["train"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    n_classes = 2
    in_feats_bbox = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats_bbox, n_classes)

    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_feats_mask,
        hidden_layer,
        n_classes,
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=0.0005
    )

    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            imgs = batch[0].to(device)

            targets = []
            for i in range(len(imgs)):
                targets.append({k: v[i].to(device) for k, v in batch[1].items()})

            print(imgs[0].shape)
            print(targets[0]["mask"].shape)
            # print(targets[0]["boxes"])

            output = model(imgs, targets)
            print(output)

            # get the average loss of the classifier, bbox predictor, mask predictor
            loss = sum(l for l in output.values())
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
