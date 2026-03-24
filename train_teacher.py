import torch
from config import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM_TEACHER,
    LR,
    NUM_EPOCHS,
)
from data import load_data
from model import CharModel
from train import train


def train_teacher():
    X, Y = load_data("names.txt", CONTEXT_SIZE)
    model = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM_TEACHER)
    train(model, X, Y, NUM_EPOCHS, BATCH_SIZE, LR)

    torch.save(model.state_dict(), "teacher.pt")


if __name__ == "__main__":
    train_teacher()
