import torch

SEQ_LENGTH = 300
N_EMBED = 400
N_HIDDEN = 256
N_OUTPUT = 1
N_LAYERS = 2
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'