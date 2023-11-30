import numpy as np
import torch

# DEVICE = "cpu"  # could not use cuda with gpu as required too much memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
BATCH_SIZE = 4
N_CLASSES = 2
LRATES = [0.001, 0.0001]
WDECAYS = [0.001, 0.0001]
SAVE_PATH = "./save_files"

np.random.seed(0)
torch.manual_seed(0)
