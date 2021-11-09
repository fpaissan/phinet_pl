from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torchvision

from torchvision import transforms
import torch.nn as nn
import torchmetrics
import torch

import cv2

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from phinet_pl.phinet import PhiNet


class PNCifar10(pl.LightningModule):
    """Lightning module for benchmarking PhiNets on CIFAR10"""
    def __init__(self):
        super().__init__()
        self.num_classes = 10

        self.model = PhiNet(
            res=32,
            alpha=0.25,
            B0=9,
            beta=1.3,
            t_zero=6,
            squeeze_excite=True, 
            h_swish=True, 
            include_top=True,
            p_l = 1
        )

        self.accuracy = torchmetrics.classification.accuracy.Accuracy()

    def forward(self, x):
        return self.model(x)

    def loss_fn(self,out,target):
        return nn.CrossEntropyLoss()(out.view(-1, self.num_classes),target)
    
    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        
        sched = CosineAnnealingLR(optimizer, T_max=100)
        lr_configs = {"scheduler": sched, 
                      "interval": "epoch", 
                      "frequency": 1, 
                      "strict": True,
                      "name": None}
        
        return [optimizer], [lr_configs]
    
    def training_step(self,batch,batch_idx):
        x, y = batch
        label = y.view(-1)
        img = x.view(-1,3,32,32)
        
        out = self(img)
        loss = self.loss_fn(out,label)
        logits = torch.argmax(out,dim=1)
        accu = self.accuracy(logits, label)
        
        self.log('train/acc', accu)
        self.log('train/loss', loss)

        return loss       

    def validation_step(self,batch,batch_idx):
        x,y = batch
        img = x.view(-1,3,32,32)
        label = y.view(-1)
        out = self(img)
        loss = self.loss_fn(out,label)
        logits = torch.argmax(out,dim=1)
        accu = self.accuracy(logits, label)

        self.log('val/loss', loss)
        self.log('val/acc', accu)

        return loss, accu
    
    def test_step(self,batch,batch_idx):
        x,y = batch
        img = x.view(-1,3,32,32)
        label = y.view(-1)
        out = self(img)
        loss = self.loss_fn(out,label)
        logits = torch.argmax(out,dim=1)
        accu = self.accuracy(logits, label)

        self.log('test/loss', loss)
        self.log('test/acc', accu)

        return loss, accu


class Cifar10Dataset(pl.LightningDataModule):
    """Data Module for CIFAR10"""
    def __init__(self, data_path, batch_size=64, transform: bool = True):
        super().__init__()
        self.batch_size = batch_size
        if transform:
            self.aug_trans = transforms.Compose([
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            self.aug_trans = transform.ToTensor()
        
        self.root_set = torchvision.datasets.CIFAR10(data_path,
                                                     train=True,
                                                     download=True, 
                                                     transform=self.aug_trans)

        self.test_set = torchvision.datasets.CIFAR10(data_path,
                                                     train=False, 
                                                     download=True, 
                                                     transform=transforms.ToTensor())

    def setup(self, stage=None):
        self.train_set, self.val_set = train_test_split(self.root_set, test_size = 0.3)

    def train_dataloader(self, stage=None):
        train_loader = DataLoader(self.train_set, batch_size=256, num_workers=16)

        return DataLoader(self.root_set, batch_size=256, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=256, num_workers=16)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=256, num_workers=16)
