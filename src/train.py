import os
import time

import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        num_epochs,
        use_cuda=True,
        output_dir='output/',
        print_every=2,
    ):
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        self.print_every = print_every
        
        model = model.to(self.device)
        self.model = model

        self.losses = {
            "train_total": [],
            "val_total": [],
        }
        self.accuracies = {
            "train_total": [],
            "val_total": [],
        }

    def train(self, train_loader, val_loader=None):
        train_start = time.time()
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
        
            total_train_loss = 0.0
            total_epoch = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                total_epoch += len(labels)
                
            train_loss_epoch = float(total_train_loss) / (total_epoch+1)
            train_acc_epoch = get_accuracy(self.model, train_loader)     
            self.losses["train_total"].append(train_loss_epoch)
            self.accuracies["train_total"].append(train_acc_epoch)

            if val_loader is not None:
                total_val_loss = 0.0
                total_epoch = 0
                self.model.eval()
                
                with torch.no_grad():
                    
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # Forward pass - compute outputs on input data using the model
                        outputs = self.model(inputs)
                        
                        # Compute loss
                        loss = self.criterion(outputs, labels)
                        total_val_loss += loss.item()
                        total_epoch += len(labels)
                        
                    val_loss_epoch = float(total_val_loss) / (total_epoch+1)
                    val_acc_epoch = get_accuracy(self.model, val_loader)     
                    self.losses["val_total"].append(val_loss_epoch)
                    self.accuracies["val_total"].append(val_acc_epoch)
      
            self.save(f"model.pt")

            if epoch % self.print_every == 0:
                print(f"Epoch {epoch}: train loss {train_loss_epoch} acc {train_acc_epoch}")
                if val_loader is not None:
                    print(f"Epoch {epoch}: val loss {val_loss_epoch} acc {val_acc_epoch}")

        train_time = int(time.time() - train_start)

        print(f"Training took {train_time} seconds")

    def save(self, file_name):
        path = self.output_dir + "/" + file_name
        if os.path.exists(self.output_dir):
            pass
        else:
            os.mkdir(self.output_dir)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def get_accuracy(model, dataloader):
    #select index with maximum prediction score
    correct = 0
    total = 0
    for data in dataloader:
        mfcc, labels = data
        output = model(mfcc)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total