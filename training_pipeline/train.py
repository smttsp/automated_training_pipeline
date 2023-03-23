import sys

import torch

import torchmetrics


EPS = sys.float_info.epsilon


def train_step(
    model,
    cur_dataloader,
    loss_fn,
    optimizer,
    # device
):
    model.train()

    total_loss = 0
    total_acc = 0

    for X, y in cur_dataloader:
        # X = X.to(device)
        # y = y.to(device)
        y_logits = model(X)
        y_preds = torch.softmax(y_logits, dim=1)
        y_pred_labels = y_preds.argmax(dim=1)
        loss = loss_fn(y_logits, y)
        acc = torchmetrics.Accuracy(y, y_pred_labels)

        total_loss += loss
        total_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cur_len = len(cur_dataloader)
    mean_train_loss = total_loss / (cur_len + EPS)
    mean_train_acc = total_acc / (cur_len + EPS)

    return mean_train_loss, mean_train_acc


def eval_step(
    model,
    cur_dataloader,
    loss_fn,
    # device,
):
    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.inference_mode():
        for X, y in cur_dataloader:
            # X = X.to(device)
            # y = y.to(device)

            y_logits = model(X)
            y_preds = torch.softmax(y_logits, dim=1)
            y_pred_labels = y_preds.argmax(dim=1)
            loss = loss_fn(y_logits, y)
            acc = torchmetrics.Accuracy(y, y_pred_labels)

            total_loss += loss
            total_acc += acc

    cur_len = len(cur_dataloader)
    mean_loss = total_loss / (cur_len + EPS)
    mean_accuracy = total_acc / (cur_len + EPS)

    return mean_loss, mean_accuracy


def train_model():
    pass

