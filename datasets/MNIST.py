import torch
from torch.utils.data import Dataset
from keras.datasets import mnist
from ..utils import label_vectoriser


class MNIST(Dataset):
    def __init__(self, train=True, scaled=False, augment=False, label_smoothing=0.0):
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        train_X = torch.reshape(torch.from_numpy(train_X), (-1, 1, 28, 28))
        test_X = torch.reshape(torch.from_numpy(test_X), (-1, 1, 28, 28))

        train_X = train_X.float()
        test_X = test_X.float()

        train_y = label_vectoriser(train_y, 10, label_smoothing)

        if train:
            self.data =  {
                'X': train_X, 
                'y': train_y, 
            }
        else:
            self.data =  {
            'X': test_X, 
            'y': test_y, 
        }

    def __len__(self):
        return len(self.data['X'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.data['X'][idx], self.data['y'][idx])

        return sample