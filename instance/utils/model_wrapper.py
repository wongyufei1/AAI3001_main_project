# Optional: To modify Faster-RCNN to return losses in eval mode
# https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn

# Evaluation code used in link below
# https://github.com/pytorch/vision/blob/main/references/detection/engine.py

import gc
import os
import time

import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from instance.utils.config import SAVE_PATH
from instance.utils.references import engine


class MRCNNModelWrapper:
    def __init__(self, model, n_classes, device="cpu", weights=None, optimizer=None, epochs=None):
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
            print(f"\nTrain Loss: {train_loss}")
            print(f"Validation Loss: {val_loss}")

            if best_loss is None or best_loss > val_loss:
                print(f"Best Epoch Validation Loss: {best_loss} -> {val_loss}\n")
                best_epoch = epoch
                best_loss = val_loss
                best_weights = self.model.state_dict()

                # save model checkpoints just in case
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)

                torch.save(best_weights, os.path.join(SAVE_PATH, "cp_model.pt"))

        return best_epoch, best_loss, best_weights

    def train(self, dataloader):
        # set model to training mode
        self.model.train()

        avg_loss = 0

        time.sleep(0.1)
        for batch in tqdm(dataloader):
            imgs = batch[0].to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items() if k != "image_id"} for t in batch[1]]

            outputs = self.model(imgs, targets)

            # get the average loss of the classifier, bbox predictor, mask predictor
            loss = sum(l for l in outputs.values())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.detach()

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
            time.sleep(0.1)
            for batch in tqdm(dataloader):
                imgs = batch[0].to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items() if k != "image_id"} for t in batch[1]]

                outputs = self.model(imgs, targets)

                # get the average loss of the classifier, bbox predictor, mask predictor
                loss = sum(l for l in outputs.values())
                avg_loss += loss.detach()

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
            del img, output
            gc.collect()
            torch.cuda.empty_cache() if self.device == "cuda" else None

        return output.to("cpu")

    def predict_batches(self, dataloader, timer=False):
        # set model to evaluation mode to get detections
        self.model.eval()

        total_time = 0

        out_imgs = []
        out_predictions = []

        # do not record computations for computing the gradient
        with torch.no_grad():
            time.sleep(0.1)
            for batch in tqdm(dataloader):
                imgs = batch[0].to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items() if k != "image_id"} for t in batch[1]]

                # sync time for gpu if device is gpu
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                else:
                    start_time = time.time()

                outputs = self.model(imgs, targets)

                # sync time for gpu if device is gpu
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                else:
                    end_time = time.time()

                total_time += end_time - start_time

                out_imgs.extend(imgs.to("cpu"))
                out_predictions.extend(outputs)

        # return model's total inference time
        if timer is True:
            return out_imgs, out_predictions, total_time
        else:
            return out_imgs, out_predictions
