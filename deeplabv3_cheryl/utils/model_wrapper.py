# Optional: To modify Faster-RCNN to return losses in eval mode
# https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn

import gc
import os
from deeplabv3_cheryl.utils.metrics_functions import iou
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch

from deeplabv3_cheryl.utils.references import engine


class SemanticModelWrapper:
    def __init__(self, model, n_classes, device="cpu", weights=None, optimizer=None, epochs=None, criterion=None):
        self.model = model
        self.device = device
        self.n_classes = n_classes
        self.weights = weights
        self.config_model(n_classes, weights)
        self.optimizer = optimizer
        self.epochs = epochs
        self.criterion = criterion

        # store losses for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []


    def config_model(self, out_classes, weights):
        # configurate last layer of deeplabv3
        self.model.classifier= DeepLabHead(960, out_classes)
        print(out_classes)

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

            train_loss, train_iou = self.train(train_loader)
            val_loss, val_iou = self.validate(val_loader)

            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            self.train_losses.append(train_loss)
            self.train_ious.append(train_iou)
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

        total_train_loss = 0.0
        total_train_iou = 0.0

        for images, masks in dataloader:
            images, masks = images.to(self.device), masks.to(self.device)
            train_outputs = self.model(images)['out']

            masks = masks.expand(-1, 2, -1, -1)

            # Assuming masks are single-channel and Long type
            loss = self.criterion(train_outputs, masks.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()  # Accumulate the loss

            total_train_iou += iou(train_outputs.cpu(), masks.cpu()).item()

        return total_train_loss / len(dataloader), total_train_iou / len(dataloader)

    def validate(self, dataloader):
        # set model to train mode to retrieve losses, but do not calculate gradients
        self.model.eval()

        total_val_loss = 0.0
        total_val_iou = 0.0

        with torch.no_grad():
            for val_images, val_masks in dataloader:
                val_images, val_masks = val_images.to(self.device), val_masks.to(self.device)
                val_outputs = self.model(val_images)['out']

                masks = val_masks.expand(-1, 2, -1, -1)

                loss = self.criterion(val_outputs, masks.float())
                total_val_loss += loss.item()  # Accumulate the loss

                total_val_iou += iou(val_outputs.cpu(), val_masks.cpu()).item()


        return total_val_loss / len(dataloader), total_val_iou / len(dataloader)

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

