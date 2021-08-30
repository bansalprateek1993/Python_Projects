from kornia.augmentation.augmentation import RandomHorizontalFlip,RandomRotation
import torch
import torch.nn as nn
from torch.nn import functional as func
import torch.optim as optim
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
import math
from argparse import ArgumentParser
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.core.lightning import LightningModule
from kornia.augmentation import RandomCrop, Normalize
from model.resnet import ResNet18

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 60):
        optim_factor = 3
    elif(epoch > 40):
        optim_factor = 2
    elif(epoch > 20):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)
    
    
class SomeCallback(Callback):
    def __init__(self, model):
        self.model = model
    def on_train_epoch_start(self, trainer,pl_module):
        if trainer.current_epoch == 21:
            print("Changing learning rate to:",learning_rate(0.1, trainer.current_epoch))
            print("Learning rate:", learning_rate(0.1, trainer.current_epoch))
            trainer.optimizers = [optim.SGD(self.model.parameters(), lr=learning_rate(0.1, trainer.current_epoch), momentum=0.9, weight_decay=5e-4)]
        elif trainer.current_epoch == 41:
            print("Changing learning rate to:",learning_rate(0.1, trainer.current_epoch))
            print("Learning rate:", learning_rate(0.1, trainer.current_epoch))
            trainer.optimizers = [optim.SGD(self.model.parameters(), lr=learning_rate(0.1, trainer.current_epoch), momentum=0.9, weight_decay=5e-4)]
        elif trainer.current_epoch == 61:
            print("Changing learning rate to:",learning_rate(0.1, trainer.current_epoch))
            print("Learning rate:", learning_rate(0.1, trainer.current_epoch))
            trainer.optimizers = [optim.SGD(self.model.parameters(), lr=learning_rate(0.1, trainer.current_epoch), momentum=0.9, weight_decay=5e-4)]

class LeNet(nn.Module):
    """Definition of Lenet

    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)

        x = func.relu(self.fc1(x))        
        x = func.relu(self.fc2(x))        
        x = self.fc3(x)        
        return x



class PL_ResNet(LightningModule):
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
        parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.01)")
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
        parser.add_argument("--optimizer")
        
        parser.add_argument("--output_size", type=int, default=200)
        parser.add_argument("--scale_factor", type=int, default=1)
        parser.add_argument("--f_bn", type=bool,default=True)
        parser.add_argument("--res_bn1", type=bool,default=True)
        parser.add_argument("--res_bn2", type=bool,default=True)
        parser.add_argument("--res_bn3", type=bool,default=True)
        return parser 

    def __init__(self, lr, momentum, optimizer, dataset, output_size, scale_factor, f_bn, res_bn1, res_bn2, res_bn3,  **kwargs):
        super(PL_ResNet, self).__init__()
        self.save_hyperparameters(
          'lr',
        #   'optimizer',
          'momentum',
          'dataset',
          'scale_factor',
          'f_bn',
          'res_bn1',
          'res_bn2',
          'res_bn3'
        )

        # Image transformation using Kornia
        self.transform_training = nn.Sequential(
            RandomCrop(size=(64,64), padding=4),
            RandomRotation(20),
            RandomHorizontalFlip(p=0.5),
            Normalize(mean=torch.Tensor([0.4914, 0.4822, 0.4465]), std=torch.Tensor([0.2023, 0.1994, 0.2010])),
        )
        
        self.transform_eval = nn.Sequential(
            RandomCrop(size=(64,64), padding=4),
            Normalize(mean=torch.Tensor([0.4914, 0.4822, 0.4465]), std=torch.Tensor([0.2023, 0.1994, 0.2010])),
        )   
        self.dataset = dataset
        self.model = ResNet18(output_size=output_size, scale_factor=scale_factor, f_bn=f_bn, res_bn=[res_bn1, res_bn2, res_bn3])

        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        
        self.metric = Accuracy()


    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch # batch, channel, w, h
        x = self.transform_training(x)
        y_hat = self(x)

        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True,  prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True,  prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]

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
