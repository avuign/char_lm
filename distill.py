import torch
import torch.nn as nn

from config import (
    ALPHA,
    BATCH_SIZE,
    CONTEXT_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM_STUDENT,
    HIDDEN_DIM_TEACHER,
    LR,
    NUM_EPOCHS,
    TEMP,
)
from data import load_data
from model import CharModel
from train import train_via_distillation


def train_student_with_distillation():
    X, Y = load_data("names.txt", CONTEXT_SIZE)

    teacher = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM_TEACHER)
    teacher.load_state_dict(torch.load("teacher.pt"))

    student = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM_STUDENT)
    train_via_distillation(
        student, X, Y, teacher, TEMP, ALPHA, NUM_EPOCHS, BATCH_SIZE, LR
    )

    torch.save(student.state_dict(), "student_distill.pt")


if __name__ == "__main__":
    train_student_with_distillation()
