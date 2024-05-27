import torch

SEQ_LENGTH = 300
N_EMBED = 100
N_HIDDEN = 256
N_OUTPUT = 3
N_LAYERS = 2
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

CONTRACTION_REPLACEMENTS = {
    "don\'t": "do not",
    "dont": "do not",
    "doesn\'t": "does not",
    "doesnt": "does not",
    "didn\'t": "did not",
    "didnt": "did not",
    "shouldn\'t": "should not",
    "shouldnt": "should not",
    "mustn\'t": "must not",
    "mustnt": "must not",
    "haven\'t": "have not",
    "hvn\'t": "have not",
    "havent": "have not",
    "hadn\'t": "had not",
    "hadnt": "had not",
    "can\'t": "can not",
    "cant": "can not",
    "cannot": "can not",
    "cann\'t": "can not",
    "couldn\'t": "could not",
    "couldnt": "could not",
    "aren\'t": "are not",
    "arent": "are not",
    "isn\'t": "is not",
    "isnt": "is not",
    "wasn\'t": "was not",
    "wasnt": "was not",
    "weren\'t": "were not",
    "werent": "were not",
    "won\'t": "will not",
    "wont": "will not",
    "wouldn\'t": "would not",
    "wouldnt": "would not",
}