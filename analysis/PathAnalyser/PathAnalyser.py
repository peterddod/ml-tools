import BinaryPathTree
import torch
from torch import nn

"""
A path analyser for neural networks. Init, then call analyse
with a pytorch dataloader and model.
TODO: add list of layer sizes as structure
"""
class PathAnalyser:
    def __init__(self):
        self.structure = None
        self.active = None
        self.inactive = None

    def get_number_of_leafs(self):
        if self.active==None and self.inactive==None:
            return 1
        else:
            def get_count(node):
                if node == None:
                    return 0
            
                return node.get_number_of_leafs()
                
            left_count = get_count(self.active)
            right_count = get_count(self.inactive)
            
            return left_count + right_count

    """
    analyse the pathways created in a neural network
    params
        f: how many steps in samples between recording number of leaf nodes (unique paths)
        n: terminate after this many samples, when None defaults to number of samples in dataset
    """
    def analyse(self, model, data_loader, n=None, f=1000, device=torch.device('cpu')):
        model.eval()

        if n==None:
            n = len(data_loader)
    
        paths = BinaryPathTree()
        leafs = []

        weight_idx = []

        for i, module in enumerate(model.modules()):
            if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
                weight_idx.append(i)
        
        i = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device).to(torch.float32), target.to(device).to(torch.float32)

                outputs = []

                for i, module in enumerate(model.modules()):
                    data = module(data)

                    if i in weight_idx:
                        outputs.append(torch.flatten(data,1))

                path = torch.hstack(outputs)
                
                path = path.squeeze()

                path[path!=0] = 1
                paths.add(list(path))
                
                i += 1
                
                if i%f==0:
                    leafs.append(self.get_number_of_leafs())
                
                if i==n:
                    break

        return leafs