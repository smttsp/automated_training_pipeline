import matplotlib.pyplot as plt
import random


def visualize_random_inputs_from_dataloader(dataset, rowcol: tuple, suptitle="training"):
    classes = dataset.classes
    targets = dataset.targets
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
