import numpy as np
import torch

DEVICE = "cpu"
EPOCHS = 3
BATCH_SIZE = 1
N_CLASSES = 2
LRATES = [0.001]  # , 0.0001
WDECAYS = [0.001]  # , 0.0001
ORIG_DATA_PATH = "../MoNuSeg/MoNuSeg 2018 Training Data"
NEW_DATA_PATH = "../data"
SAVE_PATH = "./save_files"

np.random.seed(0)
torch.manual_seed(0)
