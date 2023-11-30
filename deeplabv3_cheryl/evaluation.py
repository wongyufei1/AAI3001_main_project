import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from utils.CustomDataset import CustomDataset
import os
from utils.metrics_functions import calculate_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_folder = 'MoNuSegTestData'

# Get a list of image file names in the folder
test_image_files = [f for f in os.listdir(test_folder) if f.endswith('.tif')]
test_annotation_files = [f for f in os.listdir(test_folder) if f.endswith('.xml')]

# Create a list of full paths to the images
test_image_paths = [os.path.join(test_folder, f) for f in test_image_files]
test_annotation_paths = [os.path.join(test_folder, f) for f in test_annotation_files]


test_dataset = CustomDataset(
    image_paths=test_image_paths,
    annotation_paths=test_annotation_paths,
    mask_dir='data/test/mask'
)

val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


model = deeplabv3_mobilenet_v3_large(pretrained=True)
model.load_state_dict(torch.load('trained_deeplabv3.pth'))
criterion = nn.BCEWithLogitsLoss()

# Use the function
val_accuracy = calculate_accuracy(model, val_loader, criterion)

# Print the results
print(f'Overall Accuracy: {val_accuracy * 100:.2f}%')
