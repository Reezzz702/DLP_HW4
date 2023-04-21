import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader

from model import ResNet18, ResNet50
from dataloader import RetinopathyLoader
from tqdm import tqdm
import hydra
import matplotlib.pyplot as plt 


class Trainer():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # hyperparameters
        self.batch_size = cfg.batch_size
        self.epoch = cfg.epoch
        self.ckpt_path = cfg.ckpt

        self.train_loader, self.test_loader = self.get_dataloader()

        if cfg.model == "resnet18":
            self.model = ResNet18()
        else:
            self.model = ResNet50()

        self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def get_dataloader(self):
        train_dataloader = DataLoader(
            RetinopathyLoader("new_train", 'train'),
            batch_size=self.batch_size,
            shuffle=True
        )
        test_dataloader = DataLoader(
            RetinopathyLoader("test", 'test'),
            batch_size=self.batch_size,
            shuffle=False
        )
        return train_dataloader, test_dataloader


    def iter(self):
        acc = []
        for _, (x, label) in enumerate(tqdm(self.train_loader)):
            x.to(self.device)
        return acc


    def train(self):
        for ep in range(self.epoch):
            # train_
            self.model.train()
            train_acc = self.iter()
            self.model.eval()
            test_acc = self.iter


    def plot(self):
        pass


@hydra(hydra.main(config_path=f"config", config_name="config", version_base="1.2"))
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()