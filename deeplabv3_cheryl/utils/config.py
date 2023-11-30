import torch
import torch.optim as optim


N_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10



