import sys

import matplotlib.pyplot as plt
import torch
import torchmetrics
from torch import nn
from tqdm import tqdm

from training_pipeline.dataloader import get_dataloaders
from training_pipeline.simple_cnn import Simple_CNN_Classification
from training_pipeline.utils.metrics_utils import get_confusion_matrix


EPS = sys.float_info.epsilon


def train_step(model, cur_dataloader, loss_fn, acc_fn, optimizer, device):
    model.train()

    total_loss = 0
    total_acc = 0

    for X, y in cur_dataloader:
        X, y = X.to(device), y.to(device)

        y_logits = model(X)
        y_preds = torch.softmax(y_logits, dim=1)
        y_pred_labels = y_preds.argmax(dim=1)
        loss = loss_fn(y_logits, y)
        acc = acc_fn(y, y_pred_labels)

        total_loss += loss
        total_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cur_len = len(cur_dataloader)
    mean_train_loss = total_loss / (cur_len + EPS)
    mean_train_acc = total_acc / (cur_len + EPS)

    return mean_train_loss, mean_train_acc


def get_all_predictions(model, cur_dataloader, device):
    model.eval()
    y_pred_all = []
    y_all = []
    with torch.inference_mode():
        for X, y in cur_dataloader:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_pred = torch.softmax(y_logits, dim=1)
            y_pred_labels = y_pred.argmax(dim=1)
            y_pred_all.extend(y_pred_labels)
            y_all.extend(y)
    y_true = torch.Tensor([y.item() for y in y_all])
    y_preds = torch.Tensor([y.item() for y in y_pred_all])
    return y_true, y_preds


def eval_step(
    model,
    cur_dataloader,
    loss_fn,
    acc_fn,
    device,
):
    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.inference_mode():
        for X, y in cur_dataloader:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.softmax(y_logits, dim=1)
            y_pred_labels = y_preds.argmax(dim=1)
            loss = loss_fn(y_logits, y)
            acc = acc_fn(y, y_pred_labels)

            total_loss += loss
            total_acc += acc

    cur_len = len(cur_dataloader)
    mean_loss = total_loss / (cur_len + EPS)
    mean_accuracy = total_acc / (cur_len + EPS)

    return mean_loss, mean_accuracy


def prepare_model(train_loader, device):
    first_X, first_y = next(iter(train_loader))
    num_classes = len(train_loader.dataset.dataset.classes)

    model = Simple_CNN_Classification(first_X.shape, hidden_units=32, output_shape=num_classes).to(
        device=device
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device=device)
    return model, loss_fn, acc_fn, optimizer


def train_model(cfg, device, wandb):
    epochs = cfg.get("training", {}).get("epochs", 5)

    train_loader, val_loader, test_loader = get_dataloaders(cfg, wandb)
    model, loss_fn, acc_fn, optimizer = prepare_model(train_loader, device)
    class_names = test_loader.dataset.classes

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model,
            cur_dataloader=train_loader,
            loss_fn=loss_fn,
            acc_fn=acc_fn,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = eval_step(
            model,
            cur_dataloader=val_loader,
            acc_fn=acc_fn,
            loss_fn=loss_fn,
            device=device,
        )
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # print(
        #     f"\n{epoch=}"
        #     f"\n\tTrain      --- loss: {train_loss}, acc: {train_acc}"
        #     f"\n\tValidation --- loss: {val_loss}, acc: {val_acc}"
        # )
    test_loss, test_acc = eval_step(
        model,
        cur_dataloader=test_loader,
        acc_fn=acc_fn,
        loss_fn=loss_fn,
        device=device,
    )

    wandb.log({
        # "total_epochs": epochs,
        "test_loss": test_loss,
        "test_acc": test_acc
    })
    # print(f"\n\tTest results --- loss: {test_loss}, acc: {test_acc}")

    y_true, y_preds = get_all_predictions(model, test_loader, device)
    fig, ax = get_confusion_matrix(y_true, y_preds, class_names, device)

    return model
