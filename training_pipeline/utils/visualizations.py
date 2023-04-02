import matplotlib.pyplot as plt
import random
import numpy
from torch.utils.data.dataset import Subset


def get_classes_targets(dataset):
    if isinstance(dataset, Subset):
        classes = dataset.dataset.classes
        indices = set(dataset.indices)
        targets = [x for idx, x in enumerate(dataset.dataset.targets) if idx in indices]
    else:
        classes = dataset.classes
        targets = dataset.targets
    return classes, targets


def visualize_random_inputs(dataset, rowcol: tuple, suptitle="training"):
    # TODO (smttsp): fix this function
    classes, targets = get_classes_targets(dataset)
    rows, cols = rowcol
    samples = random.sample(range(len(targets)), rows * cols)

    plt.figure(figsize=(2 * cols, 2 * rows))
    plt.suptitle(suptitle.upper(), color="r")
    for idx, pos in enumerate(samples):
        image = dataset.data[pos]
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(image, cmap="gray")
        plt.axis(False)
        class_name = classes[targets[pos]]
        plt.title(class_name)
    plt.show()
    return None


def get_model_predictions(model, images):
    device = model.device()
    y_logits = model(images.to(device))
    top_p, top_class = y_logits.topk(1, dim=1)
    top_class = top_class.squeeze().to("cpu")
    images.to("cpu")
    return top_class


def visualize_output_images(images, labels, pred_labels, classes, row_col=(5, 5), title="training", wandb=None):
    # y_labels = get_model_predictions(model, images)
    print(labels)
    print(pred_labels)

    row, col = row_col
    size = row * col
    # assuming you have a list of 25 images named 'images'
    fig, axs = plt.subplots(col, row, figsize=(10, 10))
    axs = axs.flatten()  # flatten the 2D array of axes

    for img, label, y_label, ax in zip(images[:size], labels[:size], pred_labels[:size], axs):
        ax.imshow(img.permute(1, 2, 0).numpy())
        color = "green" if y_label == label else "red"
        ax.set_title(f"{classes[label]}", color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    wandb.log({f"output/{title} prediction visualization": wandb.Image(plt)})

    return None


def visualize_output_confusion_matrix(
    labels, pred_labels, classes, row_col=(5, 5), title="training", wandb=None
):
    # y_labels = get_model_predictions(model, images)
    print(labels)
    print(pred_labels)

    # row, col = row_col
    # size = row * col
    # assuming you have a list of 25 images named 'images'
    # fig, axs = plt.subplots(col, row, figsize=(10, 10))
    # axs = axs.flatten()  # flatten the 2D array of axes
    #
    # for img, label, y_label, ax in zip(images[:size], labels[:size], pred_labels[:size], axs):
    #     ax.imshow(img.permute(1, 2, 0).numpy())
    #     color = "green" if y_label == label else "red"
    #     ax.set_title(f"{classes[label]}", color=color)
    #     ax.axis('off')

    plt.tight_layout()
    plt.show()
    wandb.log({f"output/{title} prediction visualization": wandb.Image(plt)})

    return None



def visualize_data_distribution(dataset, title="training", wandb=None):
    classes, targets = get_classes_targets(dataset)
    hist, _ = numpy.histogram(targets, bins=len(classes))
    bins = numpy.array(range(len(hist)))
    plt.figure()
    plt.bar(bins, hist, width=0.6)

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {title}", color="r")

    # plt.show()
    wandb.log({f"input/{title} data distribution": wandb.Image(plt)})

    return None
