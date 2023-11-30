import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to calculate Intersection over Union (IoU)
def iou(predicted, target):
    predicted = predicted > 0.5  # Convert to binary: 1 if > 0.5 else 0
    target = target > 0.5  # Ensure target is also binary

    intersection = (predicted & target).float().sum()  # Intersection
    union = (predicted | target).float().sum()         # Union

    if union == 0:
        return 1  # Avoid division by 0; if both are 0, IoU is 1 by definition
    else:
        return intersection / union

def calculate_accuracy(model, data_loader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            # Ensure the model is on the same device as the input tensor
            model = model.to(device)
            outputs = model(images)['out']
            preds = torch.sigmoid(outputs) > 0.5

            # Ensure preds has the same number of channels as masks
            preds = preds[:, :1, :, :] 

            # Calculate overall accuracy
            correct_predictions += (preds == masks).sum().item()
            total_predictions += masks.numel()


    # Calculate overall accuracy
    accuracy = correct_predictions / total_predictions

    return accuracy
