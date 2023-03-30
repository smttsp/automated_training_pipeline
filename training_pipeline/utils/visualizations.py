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
