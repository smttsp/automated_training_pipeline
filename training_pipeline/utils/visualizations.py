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

    rows, cols = rowcol
    samples = random.sample(range(len(dataset)), rows * cols)
    # item = dataset.data[x[0]]

    plt.figure(figsize=(2 * cols, 2*rows))
    plt.suptitle(suptitle.upper(), color="r")
    for idx, pos in enumerate(samples):
        image = dataset.data[pos]
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(image, cmap='gray')
        plt.axis(False)
        class_name = classes[targets[pos]]
        plt.title(class_name)
    plt.show()
    pass
