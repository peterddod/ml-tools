import torch
from torch import nn
from ..modules import PathedLayer


class QSNetwork(nn.Module):
    def __init__(self, pretrained_model, verbosity=0):
        super(QSNetwork, self).__init__()

        weight_idx = []
        skip_idx = []
        current_idx = 0

        for i, module in enumerate(pretrained_model.modules()):
            if isinstance(module, (nn.Flatten, nn.LogSoftmax)):
                continue
            elif isinstance(module, (nn.Linear, nn.LazyLinear)):
                current_idx = i+1
                weight_idx.append(i)
            elif current_idx == i-1 and not isinstance(module, (nn.ReLU)):
                del weight_idx[current_idx]
            else:
                skip_idx.append(i)

        module_list = []

        for i, module in enumerate(pretrained_model.modules()):
            if i in skip_idx:
                continue

            if i in weight_idx:
                module = PathedLayer(module)

            module_list.append(module)

        self.model = nn.Sequential(*module_list)

    def forward(self, X):
        return self.model(X)