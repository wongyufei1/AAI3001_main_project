from torch.utils.data import DataLoader
import os
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from semantic_fcn.utils.CustomDataset import CustomDataset
from semantic_fcn.utils.fcn_model_wrapper import FCNModelWrapper
from semantic_fcn.utils.config import *
from semantic_fcn.utils.helper import draw_predictions

test_images_folder = '../data/test_combined/images'
test_annotations_folder = '../data/test_combined/annotations'

test_image_paths = [os.path.join(test_images_folder, f) for f in os.listdir(test_images_folder) if f.endswith('.tif')]
test_annotation_paths = [os.path.join(test_annotations_folder, f.replace('.tif', '.xml')) for f in
                         os.listdir(test_images_folder) if f.endswith('.tif')]

test_dataset = CustomDataset(
    image_paths=test_image_paths,
    annotation_paths=test_annotation_paths,
    mask_dir='data/test_combined/mask'
)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, progress=True)
model = model.to(DEVICE)

model_wrapper = FCNModelWrapper(
    model=model,
    n_classes=N_CLASSES,
    device=DEVICE,
    weights=torch.load(os.path.join(SAVE_PATH, "trained_fcn.pth"), map_location=DEVICE)
)

imgs, mask, infer_time = model_wrapper.predict_batches(dataloader=test_loader, timer=True)

print(f"\nEvaluation results for... \nData Size:{len(test_dataset)}\nBatch Size:{BATCH_SIZE}\nDevice: {DEVICE}")
print(f"Total inference time: {infer_time}")
print(f"Average inference time per batch: {infer_time / BATCH_SIZE}")

"""
    Draw and save predictions
"""
if not os.path.exists(os.path.join(SAVE_PATH, "predictions")):
    os.makedirs(os.path.join(SAVE_PATH, "predictions"))

# draw and save predictions
for idx, img in enumerate(imgs):
    mask_images = draw_predictions(img, mask)

    for i, mask_image in enumerate(mask_images):
        # Save each predicted image separately
        save_path = os.path.join(SAVE_PATH, "predictions", f"test_{idx}_mask_{i}.png")
        mask_image.save(save_path)
