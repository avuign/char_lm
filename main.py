import torch
import torch.nn as nn

from config import (
    CONTEXT_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM_STUDENT,
    HIDDEN_DIM_TEACHER,
)
from data import encoding_dic
from model import CharModel


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

    teacher = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM_TEACHER)
    teacher.load_state_dict(torch.load("teacher.pt"))

    student = CharModel(CONTEXT_SIZE, EMBEDDING_DIM, HIDDEN_DIM_STUDENT)
    student.load_state_dict(torch.load("student.pt"))

    print("Teacher results :\n")
    for i in range(0, 10):
        print(generate_new_names(teacher))

    print("\n")

    print("Student results :\n")
    for i in range(0, 10):
        print(generate_new_names(student))


if __name__ == "__main__":
    main()
