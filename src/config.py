import torch

SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 300

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 3e-4

NUM_CLASSES = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
