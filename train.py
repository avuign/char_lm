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


def train_via_distillation(model, X, Y, num_epochs, batch_size, lr):

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
