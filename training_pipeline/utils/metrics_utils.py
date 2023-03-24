import torchmetrics
from mlxtend.plotting import plot_confusion_matrix


def get_confusion_matrix(y_true, y_preds, class_names, device):
    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=len(class_names)).to(
        device
    )
    confmat_tensor = confmat(preds=y_preds, target=y_true)

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), class_names=class_names, figsize=(10, 7)
    )
    return fig, ax
