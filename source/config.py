import torch

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

SYMBOL_TO_IDX = {}
IDX_TO_SYMBOL = {}

for i, symbol in enumerate(ALPHABET):
    SYMBOL_TO_IDX[symbol] = i
    IDX_TO_SYMBOL[i] = symbol

MAXLEN = 5

PATH = "./samples"

IMAGE_SIZE = (32, 100)
BATCH_SIZE = 16
TEST_SIZE = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
