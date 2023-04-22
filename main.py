import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader

from model import ResNet18, ResNet50
from dataloader import RetinopathyLoader
from tqdm import tqdm
import hydra
import os
import numpy as np


class Trainer():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # hyperparameters
        self.batch_size = cfg.batch_size
        self.epochs = cfg.epochs
        self.ckpt_path = cfg.ckpt

        self.train_loader, self.test_loader = self.get_dataloader()

        if cfg.model == "resnet18":
            self.model = ResNet18()
        else:
            self.model = ResNet50()

        self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        os.makedirs('log', exist_ok=True)


    def get_dataloader(self):
        train_dataloader = DataLoader(
            RetinopathyLoader("new_train", 'train'),
            batch_size=self.batch_size,
            shuffle=True
        )
        test_dataloader = DataLoader(
            RetinopathyLoader("new_test", 'test'),
            batch_size=self.batch_size,
            shuffle=False
        )
        return train_dataloader, test_dataloader


    def iter(self, loader):
        total = 0
        correct = 0
        for _, (x, label) in enumerate(tqdm(loader)):
            x = x.to(device=self.device, dtype=torch.float32)
            label = label.to(device=self.device, dtype=torch.int64)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, label)
            loss.backward()
            self.optimizer.step()

            # calculate training data accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()

        return 100*correct/total


    def train(self):
        self.train_log = []
        self.test_log = []
        best_acc = 0
        for ep in range(self.epochs):
            self.model.train()
            train_acc = self.iter(self.train_loader)
            print(f'Training epoch: {ep+1} | acc: {train_acc}')
            
            self.model.eval()
            test_acc = self.iter(self.test_loader)
            print(f'Testin epoch: {ep+1} | acc: {test_acc}')

            self.train_log.append(train_acc)
            self.test_log.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), f'self.ckpt_path/best_model.pth')

        np.save(f'log/{self.cfg.model}_train.npy', np.asarray(self.train_log))
        np.save(f'log/{self.cfg.model}_test.npy', np.asarray(self.test_log))



@hydra.main(config_path="./", config_name="config", version_base="1.2")
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()