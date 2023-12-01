import torch
import torch.optim as optim


N_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
SAVE_PATH = "save_files"
BATCH_SIZE = 4

