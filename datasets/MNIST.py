import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from keras.datasets import mnist
from ..utils import label_vectoriser


class MNIST(Dataset):
    def __init__(self, train=True, one_hot_labels=True):
        self.one_hot_labels = one_hot_labels

        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        if train:
            self.data =  {
                'X': torch.reshape(torch.from_numpy(train_X), (-1, 1, 28, 28)), 
                'y': torch.from_numpy(train_y), 
            }
        else:
            self.data =  {
                'X': torch.reshape(torch.from_numpy(test_X), (-1, 1, 28, 28)), 
                'y': torch.from_numpy(test_y), 
            }

    def __len__(self):
        return len(self.data['X'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.data['X'][idx]
        y = self.data['y'][idx]

        if self.one_hot_labels:
            y = F.one_hot(y.to(torch.int64), 10)

        X, y = X.float() ,y.float()

        return (X,y)

    def train(self):
        self.one_hot_labels = True

    def eval(self):
        self.one_hot_labels = False