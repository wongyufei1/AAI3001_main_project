# Optional: To modify Faster-RCNN to return losses in eval mode
# https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn

import gc
import os

import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from modules.references import engine


class SemanticModelWrapper:
    def __init__(self, model, device="cpu", weights=None, optimizer=None, epochs=None):
        self.model = model
        self.device = device
        self.n_classes = n_classes
        self.weights = weights
        self.config_model(n_classes, weights)
        self.optimizer = optimizer
        self.epochs = epochs

        # store losses for plotting
        self.train_losses = []
        self.val_losses = []

    def config_model(self, out_classes, weights):
        # configurate last layer of bbox predictor
        in_feats_bbox = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_feats_bbox, out_classes)

        # configurate last layer of mask predictor
        in_feats_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_feats_mask,
            hidden_layer,
            out_classes,
        )

        if weights is not None:
            self.model.load_state_dict(weights)

        self.model.to(self.device)

    def fit(self, train_loader, val_loader):
        if self.epochs is None or self.optimizer is None:
            raise ValueError("Missing parameters \"epochs/optimizer\"")

        best_epoch = None
        best_loss = None

        for epoch in range(self.epochs):
            print(f"---------- Epoch {epoch}/{self.epochs - 1} ----------")

            train_loss = self.train(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"\nTrain Loss: {train_loss:<30}")
            print(f"Validation Loss: {val_loss:<30}\n")

            if best_loss is None or best_loss > val_loss:
                best_epoch = epoch
                best_loss = val_loss
                best_weights = self.model.state_dict()

        return best_epoch, best_loss, best_weights

    def train(self, dataloader):
        # set model to training mode
        self.model.train()

        avg_loss = 0

        print()
        for batch in tqdm(dataloader):
            imgs = batch[0].to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items() if k != "image_id"} for t in batch[1]]

            # print(imgs)
            # print(targets)

            outputs = self.model(imgs, targets)
            # print(outputs)

            # get the average loss of the classifier, bbox predictor, mask predictor
            loss = sum(l for l in outputs.values())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            avg_loss += loss

            # remove unused data from device to avoid OOM
            del imgs, targets, outputs, loss
            gc.collect()
            torch.cuda.empty_cache() if self.device == "cuda" else None

        return avg_loss / len(dataloader)

    def validate(self, dataloader):
        # set model to train mode to retrieve losses, but do not calculate gradients
        self.model.train()

        avg_loss = 0

        # do not record computations for computing the gradient
        with torch.no_grad():
            print()
            for batch in tqdm(dataloader):
                imgs = batch[0].to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items() if k != "image_id"} for t in batch[1]]

                outputs = self.model(imgs, targets)
                # print(outputs)

                # get the average loss of the classifier, bbox predictor, mask predictor
                loss = sum(l for l in outputs.values())
                avg_loss += loss

                # remove unused data from device to avoid OOM
                del imgs, targets, outputs, loss
                gc.collect()
                torch.cuda.empty_cache() if self.device == "cuda" else None

        return avg_loss / len(dataloader)

    def evaluate(self, dataloader):
        return engine.evaluate(self.model, dataloader, self.device)

    def predict(self, img):
        # set model to evaluation mode to get detections
        self.model.eval()

        # do not record computations for computing the gradient
        with torch.no_grad():
            img = img.to(self.device)
            output = self.model(img.unsqueeze(0))

            # remove unused data from device to avoid OOM
            del img
            gc.collect()
            torch.cuda.empty_cache() if self.device == "cuda" else None

        return output

