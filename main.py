import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import models as tv_model

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
        self.log_path = cfg.log

        self.train_loader, self.test_loader = self.get_dataloader()

        if cfg.model == "resnet18":
            if cfg.pre_train:
                self.ckpt_path += "/pre_train"
                self.log_path += "/pre_train"
                self.model = tv_model.resnet18(pretrained=True)
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, 5)
            else:
                self.model = ResNet18()
        else:
            if cfg.pre_train:
                self.ckpt_path += "/pre_train"
                self.log_path += "/pre_train"
                self.model = tv_model.resnet50(pretrained=True)
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, 5)
            else:
                self.model = ResNet50()


        self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)


    def get_dataloader(self):
        train_dataloader = DataLoader(
            RetinopathyLoader("train", 'train'),
            batch_size=self.batch_size,
            shuffle=True
        )
        test_dataloader = DataLoader(
            RetinopathyLoader("test", 'test'),
            batch_size=self.batch_size,
            shuffle=False
        )
        return train_dataloader, test_dataloader


    def iter(self, loader, mode):
        total = 0
        correct = 0
        for _, (x, label) in enumerate(tqdm(loader)):
            x = x.to(device=self.device, dtype=torch.float32)
            label = label.to(device=self.device, dtype=torch.int64)
            outputs = self.model(x)

            if mode == "train":
                self.optimizer.zero_grad()
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
            print(f"=========Training epoch {ep+1}=========")
            train_acc = self.iter(self.train_loader, "train")
            print(f'Training epoch: {ep+1} | acc: {train_acc}')
            
            self.model.eval()
            print(f"=========Testing epoch {ep+1}=========")
            test_acc = self.iter(self.test_loader, "test")
            print(f'Testing epoch: {ep+1} | acc: {test_acc}')

            self.train_log.append(train_acc)
            self.test_log.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), f'{self.ckpt_path}/{self.cfg.model}_best_model.pth')

        np.save(f'{self.log_path}/{self.cfg.model}_train.npy', np.asarray(self.train_log))
        np.save(f'{self.log_path}/{self.cfg.model}_test.npy', np.asarray(self.test_log))



@hydra.main(config_path="./", config_name="config", version_base="1.2")
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()