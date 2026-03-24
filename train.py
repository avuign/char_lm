import torch
import torch.nn as nn


def train(model, X, Y, num_epochs, batch_size, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        for i in range(0, len(X), batch_size):
            batch = {"chars": X[i : i + batch_size], "next_char": Y[i : i + batch_size]}

            logits = model.forward(torch.tensor(batch["chars"]))
            loss = nn.CrossEntropyLoss()(logits, torch.tensor(batch["next_char"]))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}, Loss: {loss}")


def distillation_loss(student_logits, teacher_logits, hard_labels, T, alpha):
    # need log-prob for student, usual prop for teacher .......
    p_student = nn.functional.log_softmax(student_logits / T, dim=-1)
    p_teacher = nn.functional.softmax(teacher_logits / T, dim=-1)
    KL_div_loss = nn.KLDivLoss(reduction="batchmean")(p_student, p_teacher)
    cross_entropy_loss = nn.CrossEntropyLoss()(student_logits, hard_labels)

    loss = alpha * cross_entropy_loss + (1 - alpha) * (T**2) * KL_div_loss

    return loss


def train_via_distillation(
    student, X, Y, teacher, T, alpha, num_epochs, batch_size, lr
):

    optimizer = torch.optim.Adam(student.parameters(), lr)

    for epoch in range(num_epochs):
        for i in range(0, len(X), batch_size):
            batch = {"chars": X[i : i + batch_size], "next_char": Y[i : i + batch_size]}

            student_logits = student.forward(torch.tensor(batch["chars"]))
            with torch.no_grad():
                teacher_logits = teacher.forward(torch.tensor(batch["chars"]))
            hard_labels = torch.tensor(batch["next_char"])

            loss = distillation_loss(
                student_logits, teacher_logits, hard_labels, T, alpha
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}, Loss: {loss}")
