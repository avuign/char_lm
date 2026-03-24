import torch
import torch.nn as nn

from data import encoding_dic, load_data
from model import CharModel
from train import train

CONTEXT_SIZE = 3
EMBEDDING_DIM = 10
NUM_EPOCHS = 10
BATCH_SIZE = 256
LR = 0.01


def generate_new_names(model):
    _, decoding = encoding_dic()
    name = [0, 0, 0]

    actual_name = ""
    while True:
        logits = model.forward(torch.tensor([name]))
        prob = nn.Softmax(dim=-1)(logits)
        sample = torch.multinomial(prob, 1).item()
        if sample == 0:
            break
        actual_name += decoding[sample]
        name = name[1:] + [sample]
    return actual_name


def main():

    X, Y = load_data("names.txt", CONTEXT_SIZE)
    model = CharModel(CONTEXT_SIZE, EMBEDDING_DIM)
    train(model, X, Y, NUM_EPOCHS, BATCH_SIZE, LR)

    for i in range(0, 10):
        print(generate_new_names(model))


if __name__ == "__main__":
    main()
