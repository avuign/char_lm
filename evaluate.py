import torch
import torch.nn as nn

from config import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM_STUDENT,
    HIDDEN_DIM_TEACHER,
)
from data import load_data
from model import CharModel


def evaluate():
    X, Y = load_data("names.txt", CONTEXT_SIZE)

    teacher = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM_TEACHER)
    student = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM_STUDENT)
    student_distill = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM_STUDENT)

    teacher.load_state_dict(torch.load("teacher.pt"))
    student.load_state_dict(torch.load("student.pt"))
    student_distill.load_state_dict(torch.load("student_distill.pt"))

    with torch.no_grad():
        logits_teacher = teacher(torch.tensor(X))
        loss_teacher = nn.CrossEntropyLoss()(logits_teacher, torch.tensor(Y))
        logits_student = student(torch.tensor(X))
        loss_student = nn.CrossEntropyLoss()(logits_student, torch.tensor(Y))
        logits_student_distill = student_distill(torch.tensor(X))
        loss_student_distill = nn.CrossEntropyLoss()(
            logits_student_distill, torch.tensor(Y)
        )

    print(f"Teacher loss: {loss_teacher.item():.4f}\n")
    print(f"Student loss: {loss_student.item():.4f}\n")
    print(f"Distilled student loss: {loss_student_distill.item():.4f}\n")


if __name__ == "__main__":
    evaluate()
