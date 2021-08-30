import torch
import torch.nn as nn
from torch.nn import functional as func
import torch.optim as optim
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from kornia.augmentation import RandomCrop, Normalize,RandomHorizontalFlip, RandomVerticalFlip,RandomAffine

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 90):
        optim_factor = 3
    elif(epoch > 60):
        optim_factor = 2
    elif(epoch > 30):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)
    
class SomeCallback(Callback):
    def __init__(self, model):
        self.model = model
    def on_train_epoch_start(self, trainer,pl_module):
        if trainer.current_epoch == 31:
            print("Changing learning rate to:",learning_rate(0.1, trainer.current_epoch))
            print("Learning rate:", learning_rate(0.1, trainer.current_epoch))
            trainer.optimizers = [optim.SGD(self.model.parameters(), lr=learning_rate(0.1, trainer.current_epoch), momentum=0.9, weight_decay=5e-4)]
        elif trainer.current_epoch == 61:
            print("Changing learning rate to:",learning_rate(0.1, trainer.current_epoch))
            print("Learning rate:", learning_rate(0.1, trainer.current_epoch))
            trainer.optimizers = [optim.SGD(self.model.parameters(), lr=learning_rate(0.1, trainer.current_epoch), momentum=0.9, weight_decay=5e-4)]
        elif trainer.current_epoch == 91:
            print("Changing learning rate to:",learning_rate(0.1, trainer.current_epoch))
            print("Learning rate:", learning_rate(0.1, trainer.current_epoch))
            trainer.optimizers = [optim.SGD(self.model.parameters(), lr=learning_rate(0.1, trainer.current_epoch), momentum=0.9, weight_decay=5e-4)]


class LeNet(nn.Module):
    """Definition of Lenet

    """
    def __init__(self, dataset):
        self.dataset = dataset
        super(LeNet, self).__init__()
        torch.manual_seed(6)
        if self.dataset == "MNIST":
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        elif self.dataset == "CIFAR":
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        elif self.dataset == "SVHN":
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)

        torch.manual_seed(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        torch.manual_seed(6)
        self.fc1 = nn.Linear(16*5*5, 120)
        
        torch.manual_seed(6)
        self.fc2 = nn.Linear(120, 84)
        
        torch.manual_seed(6)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.tanh(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.tanh(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)

        x = func.tanh(self.fc1(x))        
        x = func.tanh(self.fc2(x))        
        x = self.fc3(x)        
        return x



class PL_LeNet(LightningModule):
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
        parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.001 for Adam)")
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
        parser.add_argument("--optimizer", default="SGD")
        return parser 

    def __init__(self, lr, momentum, optimizer, dataset, **kwargs):
        super(PL_LeNet, self).__init__()
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
        elif self.dataset == "SVHN":
            self.transform_training = nn.Sequential(
                RandomCrop(size=(32, 32), padding=4),
                # RandomHorizontalFlip(p=0.5),
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )

            self.transform_eval = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )
         # Image transformation using Kornia
        self.transform_training = nn.Sequential(
            RandomCrop(size=(32,32), padding=4),
            RandomHorizontalFlip(p=0.25), 
            RandomVerticalFlip(p=0.25),
            RandomAffine(p=0.25, degrees = 0, translate=(1/16,1/16)),
            Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        )
        
        self.transform_eval = nn.Sequential(
             RandomCrop(size=(32,32), padding=4),
             Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        )   

        self.cnn = LeNet(self.dataset)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.metric = Accuracy()


    def forward(self, x):
        x = self.cnn(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch # batch, channel, w, h
        x = self.transform_training(x)
        y_hat = self(x)
        
#        print(list(model.parameters())[0])
        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True,  prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True,  prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            return optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'Adam':
            return optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSProp':
            return optim.RMSprop(self.parameters(), lr=self.lr)

    def on_epoch_start(self):
        print('\n')
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform_eval(x)
        y_hat = self(x)
        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True)
        
    def on_epoch_start(self):
        print('\n')

    def test_step(self, batch, batch_idx):
        x, y = batch
#        x = self.transform_eval(x)
        y_hat = self(x)
        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)