import torch
from config import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM_STUDENT,
    LR,
    NUM_EPOCHS,
)
from data import load_data
from model import CharModel
from train import train


def train_student():
    X, Y = load_data("names.txt", CONTEXT_SIZE)
    model = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM_STUDENT)
    train(model, X, Y, NUM_EPOCHS, BATCH_SIZE, LR)

    torch.save(model.state_dict(), "student.pt")


if __name__ == "__main__":
    train_student()
