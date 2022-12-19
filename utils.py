import torch


def label_vectoriser(labels, classes, label_smoothing=0.1):
    n = len(labels)

    y = torch.zeros([n,classes])

    for i in range(len(y)):
        for j in range(len(y[i])):
            onehot = 0
            if j == labels[i]:
                onehot = 1

            y[i,j] = (1-label_smoothing)*onehot + label_smoothing/classes

    return y