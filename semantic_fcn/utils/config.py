import numpy as np
import torch


N_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
SAVE_PATH = "save_files"
BATCH_SIZE = 4

np.random.seed(0)
torch.manual_seed(0)
