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
        self.loss = loss
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

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        start_time = tm.time()
        running_loss = .0

        for idx, (train_x, train_label) in enumerate(train_loader):
            self.optimiser.zero_grad()
            train_label = F.one_hot(train_label.to(torch.int64), self.out_classes)

            predict_y = self.model(train_x)
            loss = self.loss(predict_y, train_label)

            if idx % 200 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))

            running_loss += loss.sum().item()
            loss.backward()
            self.optimiser.step()

        end_time = tm.time()
        train_loss = running_loss/len(train_loader)

        self.log('train', epoch, train_loss, None, end_time-start_time)

        if self.custom_log_function != None:
            self.custom_log_function(self.model, self.train_log, 'train', epoch)

    def evaluate_epoch(self, loader, epoch, dataset_type):
        self.model.eval()
        
        correct = 0
        _sum = 0
        running_loss = .0
        start_time = tm.time()

        for idx, (test_x, test_label) in enumerate(loader):
            predict_y = self.model(test_x).detach()
            predict_ys = torch.argmax(predict_y, axis=1)

            running_loss += self.loss(predict_y, F.one_hot(test_label.to(torch.int64),self.out_classes))

            _ = predict_ys == test_label
            correct += torch.sum(_, axis=-1)
            _sum += _.shape[0]

        end_time = tm.time()
        accuracy = correct / _sum
        test_loss = running_loss / len(loader)

        self.log(dataset_type, epoch, test_loss, accuracy, end_time-start_time)

        if self.custom_log_function != None:
            self.custom_log_function(self.model, self.train_log, 'test', epoch)

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