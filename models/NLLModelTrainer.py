import torch
import pandas as pd
import time as tm
from torch.nn import functional as F

"""
Wrapper for PyTorch models that includes saving and
loading, training functions, evaluation functions,
and training logs. Any optimiser can be used, but
loss is limited to only NLL.
TODO: Add custom log function capability
"""
class NLLModelTrainer:
    def __init__(self, model, optimiser, device=torch.device('cpu'), custom_log_function=None, custom_log_columns=[]):
        self.model = model
        self.optimiser = optimiser
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

    def log(self, dataset_type, epoch, loss, accuracy, time):
        # check if epoch is in log, add if not
        if epoch not in self.train_log.index:
            columns = list(self.train_log.columns.values)
            entry = [.0] * len(columns)
            self.train_log.loc[epoch] = entry

        # update row
        if loss!=None and self.train_log.at[epoch,f'{dataset_type}_loss']==.0:
            self.train_log.at[epoch,f'{dataset_type}_loss'] = loss
        if accuracy!=None and self.train_log.at[epoch,f'{dataset_type}_accuracy']==.0:
            self.train_log.at[epoch,f'{dataset_type}_accuracy'] = accuracy
        if time!=None and self.train_log.at[epoch,f'{dataset_type}_time']==.0:
            self.train_log.at[epoch,f'{dataset_type}_time'] = time

    def train_epoch(self, dataloader, epoch):
        self.model.train()

        correct = 0
        total = 0
        train_loss = 0

        start_time = tm.time()

        print()

        for batch_idx, (data, target) in enumerate(dataloader):
            self.optimiser.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimiser.step()

            total += data.size(0)
            correct += (output.argmax(dim=1) == target).float().sum()
            train_loss += loss.item() * data.size(0)

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        end_time = tm.time()
        train_loss /= total
        accuracy = (correct/total).item()

        print(f'\nTrain Results - Loss: {train_loss}, Accuracy: {accuracy}')

        self.log('train', epoch, train_loss, accuracy, end_time-start_time)

    def evaluate_epoch(self, dataloader, epoch):
        self.model.eval()

        correct = 0
        total = 0
        test_loss = 0

        start_time = tm.time()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                output = self.model(data)
                loss = F.nll_loss(output, target)

                total += data.size(0)
                correct += (output.argmax(dim=1) == target).float().sum()
                test_loss += loss.item() * data.size(0)

        end_time = tm.time()
        test_loss /= total
        accuracy = (correct/total).item()

        print(f'Eval Results - Loss: {test_loss}, Accuracy: {accuracy}')

        self.log('test', epoch, test_loss, accuracy, end_time-start_time)

    def fit(self, train_loader, test_loader=None, epochs=5):
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)

            if test_loader != None:
                self.evaluate_epoch(test_loader, epoch)    
    
    def load_model(self, path):
        self.model = torch.load(path)
    
    def save_model(self, path):
        torch.save(self.model, path)

    def get_log(self):
        return self.train_log