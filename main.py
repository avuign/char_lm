import torch
import torch.nn as nn

from config import BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM, LR, NUM_EPOCHS
from data import encoding_dic, load_data
from model import CharModel
from train import train


def generate_new_names(model):
    _, decoding = encoding_dic()
    name = [0] * CONTEXT_SIZE

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
    model = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
    train(model, X, Y, NUM_EPOCHS, BATCH_SIZE, LR)

    for i in range(0, 10):
        print(generate_new_names(model))


if __name__ == "__main__":
    main()
