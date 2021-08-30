import torch
import torch.nn as nn
from torch.nn import functional as func
import torch.optim as optim
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.core.lightning import LightningModule
from kornia.augmentation import RandomCrop, Normalize


class PL_Basic(LightningModule):
    """NEtwork definition using Pytorch-Lightining and requires to define several funtions (please read the documentation).

    Args:
        lr (float): learning rate
        momemtum (float): momentum for SGD (optimizer) if it is defined
        optimizer (str): type of optimizer for training
        dataset (str): input dataset for training 
        **kwargs: other parameters 
    Returns:
        LightingModule: network based on PL ready for training and evaluation modes 
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
        parser.add_argument("--optimizer", default="SGD")
        return parser 

    def __init__(self, lr, momentum, optimizer, dataset, **kwargs):
        super(PL_Basic, self).__init__()
        self.save_hyperparameters(
          'lr',
          'optimizer',
          'momentum',
          'dataset'
        )
        self.dataset = dataset
        # Image transformation using Kornia
        if self.dataset == "MNIST":
            self.transform_training = nn.Sequential(
                RandomCrop(size=(32, 32), padding=4),
                # RandomHorizontalFlip(p=0.5),
                Normalize(mean=torch.Tensor([0.485]), std=torch.Tensor([0.229])),
            )

            self.transform_eval = nn.Sequential(
                RandomCrop(size=(32, 32), padding=4),
                # RandomHorizontalFlip(p=0.5),

                Normalize(mean=torch.Tensor([0.485]), std=torch.Tensor([0.229])),
            )
        elif self.dataset == "CIFAR":
            self.transform_training = nn.Sequential(
                RandomCrop(size=(32, 32), padding=4),
                # RandomHorizontalFlip(p=0.5),
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )

            self.transform_eval = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )
        elif self.dataset == "SVHN":
            self.transform_training = nn.Sequential(
                RandomCrop(size=(32, 32), padding=4),
                # RandomHorizontalFlip(p=0.5),
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )

            self.transform_eval = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )

        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.metric = Accuracy()
        
        torch.manual_seed(6)
        if self.dataset == "MNIST":
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        elif self.dataset == "CIFAR":
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        elif self.dataset == "SVHN":
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
#        self.l2 = nn.Linear(4,10)
        torch.manual_seed(6)
        self.fc1 = nn.Linear(6*7*7, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch  # batch, channel, w, h
        x = self.transform_training(x)
        y_hat = self(x)
#        print(list(self.parameters())[0])
        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
#        print(list(self.parameters()))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            return optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'Adam':
            return optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSProp':
            return optim.RMSprop(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform_eval(x)
        y_hat = self(x)
        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform_eval(x)
        y_hat = self(x)
        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)