import torch

def iou(predicted, target):
    predicted = predicted > 0.5  # Convert to binary: 1 if > 0.5 else 0
    target = target > 0.5  # Ensure target is also binary

    intersection = (predicted & target).float().sum()  # Intersection
    union = (predicted | target).float().sum()         # Union

    if union == 0:
        return 1  # Avoid division by 0; if both are 0, IoU is 1 by definition
    else:
        return intersection / union