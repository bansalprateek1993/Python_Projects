'''Inspired by ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

source code from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''


from argparse import ArgumentParser
from kornia.augmentation import RandomCrop, RandomHorizontalFlip, Normalize
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from torch.nn.modules.batchnorm import BatchNorm2d
from model.renet import ReNet
# from pl_renet import ReNet
from pytorch_lightning.metrics import Accuracy
import torch.optim as optim
import math


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


class BasicBlock_ReNet(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, kernel_size=(1,1), res_bn=[True, True, True] ):
        super(BasicBlock_ReNet, self).__init__()
        self.f_bn1 = res_bn[0]
        self.f_bn2 = res_bn[1]
        self.f_bn3 = res_bn[2]

        self.renet_1 = ReNet(in_planes, out_planes, kernel_size=kernel_size)
        
        if self.f_bn1:
            self.bn_1 = nn.BatchNorm2d(2 * out_planes)

        self.renet_2 = ReNet(2 * out_planes, out_planes, kernel_size=(1,1))
        
        if self.f_bn2:
            self.bn_2 = nn.BatchNorm2d(2 * out_planes)

        self.shortcut = nn.Sequential()
        if kernel_size != (1,1):
            if self.f_bn3:
                self.shortcut = nn.Sequential(
                    ReNet(in_planes, out_planes, kernel_size=kernel_size),
                    nn.BatchNorm2d(2 * out_planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    ReNet(in_planes, out_planes, kernel_size=kernel_size)
                )

    def forward(self, x):
        out = self.renet_1(x)

        if self.f_bn1:
            out = self.bn_1(out)
        
        out = F.relu(out)

        # out = F.relu(self.bn_1(self.renet_1(x)))

        out = self.renet_2(out)
        if self.f_bn2:
            out = self.bn_2(out)

        # out = self.bn_2(self.renet_2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ReNet_ResNet(nn.Module):
    def __init__(self, num_blocks, output_size=10, scale_factor=1, f_bn=True, res_bn=[True, True, True]):
        super(ReNet_ResNet, self).__init__()
        self.f_bn = f_bn
        self.in_planes = 64 // scale_factor

        self.renet_1 = ReNet(1 * 1 * 3, 32 // scale_factor, kernel_size=(1, 1))
        # self.renet_1 = ReNet(1 * 1 * 4, 32 // scale_factor, kernel_size=(1, 1))

        if self.f_bn:
            self.bn_1 = nn.BatchNorm2d(64 // scale_factor)

        self.layer_1 = self._make_layer(32 // scale_factor, num_blocks[0], kernel_size=(1, 1), res_bn=res_bn)
        self.layer_2 = self._make_layer(64 // scale_factor, num_blocks[1], kernel_size=(2, 2), res_bn=res_bn)
        self.layer_3 = self._make_layer(128 // scale_factor, num_blocks[2], kernel_size=(2, 2), res_bn=res_bn)
        self.layer_4 = self._make_layer(256 // scale_factor, num_blocks[3], kernel_size=(2, 2), res_bn=res_bn)
        self.linear = nn.Linear(4 * (512 // scale_factor), output_size)

    def _make_layer(self, planes, num_blocks, kernel_size, res_bn):
        lst_kernels = [kernel_size] + [(1, 1)] * (num_blocks - 1)
        layers = []
        for each_kernel in lst_kernels:
            layers.append(BasicBlock_ReNet(each_kernel[0] * each_kernel[1] * self.in_planes, planes, each_kernel, res_bn))
            self.in_planes = planes * 2  
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.renet_1(x)

        if self.f_bn:
            out = self.bn_1(out)

        out = F.relu(out)
        # out = F.relu(self.bn_1(self.renet_1(x)))
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ReNet_ResNet18(output_size=10, scale_factor=1, f_bn=True, res_bn=[True, True, True]):
    return ReNet_ResNet([2, 2, 2, 2], output_size=output_size, scale_factor=scale_factor, f_bn=f_bn, res_bn=res_bn)

def test_renet():
    net = ReNet_ResNet([2, 2, 2, 2], output_size=10, 
                       f_bn=False, res_bn=[False, False, False])
    y = net(torch.randn(10, 3, 64, 64))
    target = torch.tensor([1,2,3,4,5,6,7,8,9,0])
    loss = nn.CrossEntropyLoss()
    error = loss(y,target)
    error.backward()

    print(y.size())


class PL_ReNet_ResNet(LightningModule):
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
        
        parser.add_argument("--output_size", type=int, default=200)
        parser.add_argument("--scale_factor", type=int, default=1)
        parser.add_argument("--f_bn", type=bool,default=True)
        parser.add_argument("--res_bn1", type=bool,default=True)
        parser.add_argument("--res_bn2", type=bool,default=True)
        parser.add_argument("--res_bn3", type=bool,default=True)
      # parser.add_argument("--optimizer")

        return parser 

    def __init__(self, lr, momentum, dataset, output_size, scale_factor, f_bn, res_bn1, res_bn2, res_bn3, **kwargs):
        super(PL_ReNet_ResNet, self).__init__()
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
            RandomHorizontalFlip(p=0.5),
            Normalize(mean=torch.Tensor([0.4914, 0.4822, 0.4465]), std=torch.Tensor([0.2023, 0.1994, 0.2010])),
        )
        
        self.transform_eval = nn.Sequential(
            RandomCrop(size=(64,64), padding=4),
            Normalize(mean=torch.Tensor([0.4914, 0.4822, 0.4465]), std=torch.Tensor([0.2023, 0.1994, 0.2010])),
        )   
        self.dataset = dataset
        self.model = ReNet_ResNet18(output_size=output_size, scale_factor=scale_factor, f_bn=f_bn, res_bn=[res_bn1, res_bn2, res_bn3])

        self.lr = lr
        self.momentum = momentum
        
        self.metric = Accuracy()


    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=False):
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform_eval(x)
        # x = self.tokenize(x, device=self.device)
        y_hat = self(x)
        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.transform_eval(x)
        # x = self.tokenize(x, device=self.device)
        y_hat = self(x)
        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)


    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=5e-4)
#        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return optimizer
        
if __name__ == '__main__':
    test_renet()