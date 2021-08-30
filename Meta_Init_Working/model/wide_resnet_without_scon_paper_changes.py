import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.core.lightning import LightningModule
from kornia.augmentation import RandomCrop, Normalize, RandomHorizontalFlip
import math

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        
def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight)
        init.constant_(m.bias, 0)

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)
    
class SomeCallback(Callback):
    def __init__(self, model):
        self.model = model
    def on_train_epoch_start(self, trainer,pl_module):
        if trainer.current_epoch == 61:
            print("Changing learning rate to:",learning_rate(0.1, trainer.current_epoch))
            print("Learning rate:", learning_rate(0.1, trainer.current_epoch))
            trainer.optimizers = [optim.SGD(self.model.parameters(), lr=learning_rate(0.1, trainer.current_epoch), momentum=0.9, weight_decay=5e-4)]
        elif trainer.current_epoch == 121:
            print("Changing learning rate to:",learning_rate(0.1, trainer.current_epoch))
            print("Learning rate:", learning_rate(0.1, trainer.current_epoch))
            trainer.optimizers = [optim.SGD(self.model.parameters(), lr=learning_rate(0.1, trainer.current_epoch), momentum=0.9, weight_decay=5e-4)]
        elif trainer.current_epoch == 161:
            print("Changing learning rate to:",learning_rate(0.1, trainer.current_epoch))
            print("Learning rate:", learning_rate(0.1, trainer.current_epoch))
            trainer.optimizers = [optim.SGD(self.model.parameters(), lr=learning_rate(0.1, trainer.current_epoch), momentum=0.9, weight_decay=5e-4)]

##Addding mixup with alpha 1
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        
        ## Adding scale for fixup implementation for scalar multiplier initialized at 1 every two layers
        ## https://github.com/hongyi-zhang/Fixup/blob/master/cifar/models/fixup_resnet_cifar.py
#        self.scale = nn.Parameter(torch.ones(1)) 
        self.beta = nn.Parameter(torch.zeros(1))
        
    def swish_beta(self,x):
        return x*F.sigmoid(self.beta*x)

#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_planes != planes:
#            self.shortcut = nn.Sequential(
#                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
#            )

    def forward(self, x):
#       out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.dropout(self.conv1(self.swish_beta(self.bn1(x))))
#        out = self.conv2(F.relu(out))
      ##Adding code for scalar multiplier
        out = self.conv2(self.swish_beta(self.bn2(out)))
#        out = F.relu(out)
#        out += F.relu(self.shortcut(x))

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.beta = nn.Parameter(torch.zeros(1))
        
    def swish_beta(self,x):
        return x*F.sigmoid(self.beta*x)


    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
#        out = F.relu(self.bn1(out))
        out = self.swish_beta(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class PL_Wide_Resnet(LightningModule):
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
        parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.01 for SGD)")
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
        parser.add_argument("--depth", default=28, type=int, help="Number of conv layers in ResNet")
        parser.add_argument("--width", default=10, type=int, help="Width of model")
        parser.add_argument("--dropout", default=0.2, type=int, help="dropout_rate")
        parser.add_argument("--optimizer", default="SGD")
        return parser

    def __init__(self, lr, momentum, optimizer, dataset, depth, width, dropout_rate, **kwargs):
        super(PL_Wide_Resnet, self).__init__()
        self.save_hyperparameters(
            'lr',
            'optimizer',
            'momentum',
            'dataset'
        )

        self.dataset = dataset
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.metric = Accuracy()
        self.depth = depth
        self.width = width
        self.dropout_rate = dropout_rate
        self.epoch = -1
        self.cnn = Wide_ResNet(self.depth, self.width, self.dropout_rate, 10)
        self.best_acc = 0


        # Image transformation using Kornia
        if self.dataset == "CIFAR":
            self.transform_training = nn.Sequential(
#                RandomCrop(size=(32, 32), padding=4),
#                RandomHorizontalFlip(),
                Normalize(mean=torch.Tensor([0.4914, 0.4822, 0.4465]), std=torch.Tensor([0.2023, 0.1994, 0.2010])),
            )

            self.transform_eval = nn.Sequential(
                Normalize(mean=torch.Tensor([0.4914, 0.4822, 0.4465]), std=torch.Tensor([0.2023, 0.1994, 0.2010])),
            )

      
    def forward(self, x):
        x = self.cnn(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch  # batch, channel, w, h
        x = self.transform_training(x)
        
        inputs, targets_a, targets_b, lam = mixup_data(x, y)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        
        outputs = self(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)                                         
        #y_hat = self(x)

        #        print(list(model.parameters())[0])
        accuracy = self.metric(outputs, y)
        # loss = nn.CrossEntropyLoss()(y_hat, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            print("Learning Rate = ", learning_rate(self.lr, self.epoch))
            print("Epoch Number=", self.epoch)
            return optim.SGD(self.parameters(), lr=learning_rate(self.lr, self.epoch), momentum=self.momentum, weight_decay=5e-4)
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
        if(accuracy >  self.best_acc):
          self.best_accuracy = accuracy
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def on_epoch_start(self):
        self.epoch = self.epoch + 1
        print('\n')

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform_eval(x)
        y_hat = self(x)
        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)

if __name__ == '__main__':
   net=Wide_ResNet(28, 10, 0.3, 10)
   net.apply(conv_init)
   y = net(Variable(torch.randn(1,3,32,32)))
   print(y.size())