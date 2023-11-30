import numpy as np
import torch

DEVICE = "cpu"
EPOCHS = 3
BATCH_SIZE = 2
N_CLASSES = 2
LRATES = [0.001]  # , 0.0001
WDECAYS = [0.001]  # , 0.0001
SAVE_PATH = "./save_files"

np.random.seed(0)
torch.manual_seed(0)
