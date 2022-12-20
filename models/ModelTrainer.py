import torch
import pandas as pd
import time as tm
from torch.nn import functional as F

"""
Wrapper for PyTorch models that includes saving and
loading, training functions, evaluation functions,
and training logs
"""
class ModelTrainer:
    def __init__(self, model, optimiser, loss, device=torch.device('cpu'), out_classes=10, custom_log_function=None, custom_log_columns=[]):
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss
        self.device = device

        self.out_classes = 10

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
            self.train_log.loc[epoch] = entry

        # update row
        if loss!=None and self.train_log.at[epoch,f'{type}_loss']==.0:
            self.train_log.at[epoch,f'{type}_loss'] = loss
        if accuracy!=None and self.train_log.at[epoch,f'{type}_accuracy']==.0:
            self.train_log.at[epoch,f'{type}_accuracy'] = accuracy
        if time!=None and self.train_log.at[epoch,f'{type}_time']==.0:
            self.train_log.at[epoch,f'{type}_time'] = time

    def train_epoch(self, dataloader, epoch):
        self.model.train()

        running_loss = .0
        _sum = 0

        start_time = tm.time()

        for idx, sample in enumerate(dataloader):
            inputs, labels = sample[:2]
            self.optimiser.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimiser.step()

            running_loss += loss.item() * inputs.shape[0]
            _sum += inputs.shape[0]

            if idx % 200 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))

        end_time = tm.time()
        train_loss = running_loss/_sum

        self.log('train', epoch, train_loss, None, end_time-start_time)

    def evaluate_epoch(self, dataloader, epoch, dataset_type):
        self.model.eval()
        
        correct = 0
        total = 0
        running_loss = .0

        start_time = tm.time()

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in dataloader:
                images, labels_oh, labels = data
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.loss_fn(images, labels_oh)
                running_loss += loss.item() * labels.size()

        end_time = tm.time()
        test_loss = running_loss/total
        accuracy = 100 * correct/total

        print(f'{dataset_type} evaluation - loss:{test_loss}, accuracy:{accuracy}')

        self.log(dataset_type, epoch, test_loss, accuracy, end_time-start_time)

    def fit(self, train_loader, test_loader=None, epochs=5):
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            self.evaluate_epoch(train_loader, epoch, 'train')  

            if test_loader != None:
                self.evaluate_epoch(test_loader, epoch, 'test')    
    
    def load_model(self, path):
        self.model = torch.load(path)
    
    def save_model(self, path):
        torch.save(self.model, path)

    def get_log(self):
        return self.train_log