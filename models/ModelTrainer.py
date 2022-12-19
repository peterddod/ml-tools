import torch
import pandas as pd
import time as tm

"""
Wrapper for PyTorch models that includes saving and
loading, training functions, evaluation functions,
and training logs
"""
class ModelTrainer():
    def _init_(self, model, optimiser, loss, device='cpu', custom_log_function=None, custom_log_columns=[]):
        self.model = model
        self.optimiser = optimiser
        self.loss = loss
        self.device = device

        self.custom_log_function = custom_log_function
        self.train_log = pd.DataFrame(
            columns=[
                'train_loss', 'train_accuracy', 'train_time',
                'test_loss', 'test_accuracy', 'test_time',
                ] + custom_log_columns
        )

    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        return self.model(X)

    def log(self, type, epoch, loss, accuracy, time):
        # check if epoch is in log, add if not
        if epoch not in self.train_log.index:
            columns = list(self.train_log.columns.values)
            entry = [.0] * len(columns)
            self.train_log.iloc[epoch] = entry

        # update row
        self.train_log.at[epoch,f'{type}_loss'] = loss
        self.train_log.at[epoch,f'{type}_accuracy'] = accuracy
        self.train_log.at[epoch,f'{type}_time'] = time

    def train_epoch(self, train_loader, epoch):
        self.model.train()

        train_loss = 0
        correct = 0

        start_time = tm.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimiser.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimiser.step()

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                    
        end_time = tm.time()
        train_loss /= len(train_loader.dataset)
        correct /= len(train_loader.dataset)

        self.log('train', epoch, train_loss, correct, end_time-start_time)

        if self.custom_log_function != None:
            self.custom_log_function(self.model, self.train_log, 'train', epoch)

    def evaluate_epoch(self, test_loader, epoch):
        self.model.eval()

        test_loss = 0
        correct = 0

        start_time = tm.time()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss(output, target).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        end_time = tm.time()
        test_loss /= len(test_loader.dataset)

        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            set, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.log('train', epoch, test_loss, correct, end_time-start_time)

        if self.custom_log_function != None:
            self.custom_log_function(self.model, self.train_log, 'test', epoch)

    def train(self, train_loader, test_loader=None, epochs=5):
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)

            if test_loader != None:
                self.evaluate_epoch(test_loader)      
    
    def load_model(self, path):
        self.model = torch.load(path)
    
    def save_model(self, path):
        torch.save(self.model, path)

    def get_log(self):
        return self.train_log