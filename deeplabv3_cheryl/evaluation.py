import os
import torch
import torch.nn as nn
from utils.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from utils.CustomDataset import CustomDataset
from utils.fcn_model_wrapper import FCNModelWrapper
from utils.config import *
from utils.metrics_functions import calculate_accuracy, iou


def evaluate_model(model, dataloader):
    model.eval()
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data[0].to(DEVICE), data[1].to(DEVICE)

            outputs = model(inputs)['out']
            predictions = torch.argmax(outputs, dim=1)

            total_iou += iou(predictions, targets)
            total_samples += inputs.size(0)

    mean_iou = total_iou / total_samples
    return mean_iou


if __name__ == "__main__":

    test_splits = ["test_in_train", "test_not_in_train", "test_combined"]

    for split in test_splits:
        print(f"Evaluating model...(Set:{split} Metric:mIoU)")
        # Create a list of full paths to the images
        test_images_folder = f"../data/{split}/images"
        test_annotations_folder = f"../data/{split}/annotations"

        test_image_paths = [os.path.join(test_images_folder, f) for f in os.listdir(test_images_folder) if f.endswith('.tif')]
        test_annotation_paths = [os.path.join(test_annotations_folder, f.replace('.tif', '.xml')) for f in os.listdir(test_images_folder) if f.endswith('.tif')]


        test_dataset = CustomDataset(
            image_paths=test_image_paths,
            annotation_paths=test_annotation_paths,
            mask_dir=os.path.join(SAVE_PATH, "mask")
        )

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, progress=True)
        # Define and load your FCN model
        model_wrapper = FCNModelWrapper(
            model=model,
            n_classes=N_CLASSES,
            device=DEVICE,
            weights=torch.load(os.path.join(SAVE_PATH, "trained_fcn.pth"), map_location=DEVICE)
        )

        mean_iou = evaluate_model(model_wrapper.model, test_dataloader)
        test_accuracy = calculate_accuracy(model_wrapper.model, test_dataloader, model_wrapper.criterion)

        # Print the results
        print(f'Overall Accuracy: {test_accuracy * 100:.2f}%')
        print(f'Mean IoU: {mean_iou:.4f}')

        # Write the results into a text file
        result_file_path = f"evaluation_results_{split}.txt"
        with open(result_file_path, 'w') as result_file:
            result_file.write(f'Overall Accuracy: {test_accuracy * 100:.2f}%\n')
            result_file.write(f'Mean IoU: {mean_iou:.4f}\n')

        print(f'Results written to: {result_file_path}')

