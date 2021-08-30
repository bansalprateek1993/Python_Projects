import torch
import torch.nn as nn
from torch.nn import functional as func
import torch.optim as optim
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.core.lightning import LightningModule
from kornia.augmentation import RandomCrop, Normalize,RandomHorizontalFlip, RandomVerticalFlip,RandomAffine, RandomResizedCrop
from einops import rearrange
import math

'''
ReNet Architecture:
hidden size of Renet is 128.
Input size for 1st rnn i.e. lstm_h is 12 = 3(nc) * 2(Kernel_width) * 2(kernel_height)
=> Each RNN block will produce a hidden_unit of size 128,
RNN will give 2 outputs, (1) All hidden states associated with a sequence (2) Just the very last hidden state for a sequence
(1) total hidden states =  [batch_passed_to_RNN * sequence_length * hidden_unit_size] = [16 * 2048 * 256]


batch_passed_to_RNN = 16
sequence_lenth = 2048
number of features per sequence point = 12

How?
Image size including batch in tensor is (128,3,32,32)
Rearranged to (128,3,32,32) -> (16,2048,12) i.e. (32/2, 32/2 * 128, 2*2*3), here 2*2 is the kernel size.

----image horizantal--
hidden size of Renet is 128.
Input size for 1st rnn i.e. lstm_h is 12 = 3(nc) * 2(Kernel_width) * 2(kernel_height)

rnn method is to this tensor i.e. rnn(16,2048,12)  i.e. batch_passed=16, sequence_length=2048 and number_of_feature_per_sequence_point=12  
then this will give output total_hidden_states, it will be of size [16, 2048, 256], 256(2 * 128(hidden unit size)) because of bidirectional.
----

--- image vertical---
hidden unit of rnn is 128
input size of 2nd rnn is = 2 * 2 * 256

input tensor to rnn is [16,2048,256] i.e. 16 batches, 2048 sequence_length and 256 features_per_sequence_point. 
this will give output [16, 2048, 256] same as before
-------------------

we will rearrange this to [128,256,16,16],  pass into one more RNN module. i.e. passing 128 batches of 256(nc) and 16*16(image size) 
''' 

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


def weights_init(m):
    # Code taken from https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/8
    parameters = m.state_dict()
    for each_key in parameters.keys():
        print(f'Init-{each_key}')
        if 'weight_ih' in each_key:
            nn.init.orthogonal_(parameters[each_key])
        elif 'weight_hh' in each_key:
            nn.init.orthogonal_(parameters[each_key])
        elif 'bias' in each_key:
            nn.init.constant_(parameters[each_key], val=0)

def bias_init(m):
    # Code taken from https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/8
    parameters = m.state_dict()
    for each_key in parameters.keys():
        print(f'Init-{each_key}')
        if 'bias_ih_l' in each_key:
            b_ir, b_iz, b_in = parameters[each_key].chunk(3, 0)
            nn.init.constant_(b_iz, val=-1)
        elif 'bias_hh_l' in each_key:
            b_ir, b_iz, b_in = parameters[each_key].chunk(3, 0)
            nn.init.constant_(b_iz, val=-1)
            
class ReNet(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=(2, 2), rnn='GRU', depth=(1,1)):
        super(ReNet, self).__init__()
        if rnn == 'GRU':
            rnn = nn.GRU
        elif rnn == 'LSTM':
            rnn = nn.LSTM
        
        self.lstm_h = rnn(input_size, hidden_size, bias=True, num_layers=depth[0], bidirectional=True)
        self.lstm_v = rnn(hidden_size * 2, hidden_size, bias=True, num_layers=depth[1], bidirectional=True)

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
#        self.lstm_h.apply(bias_init)
#        self.lstm_v.apply(bias_init)

    def forward(self, x):
        k_w, k_h = self.kernel_size
#        print("kernel width", k_w)
#        print("kernel Height", k_h)
#        print("Size of x", x.size())
        b, c, h, w = x.size()
        assert h % k_h == 0 and w % k_w == 0, 'input size does not match with kernel size'
        x = rearrange(x, 'b c (h1 h2) (w1 w2) -> h1 (b w1) (c h2 w2)', w2=k_w, h2=k_h)
#        print(x.size())
        x, _ = self.lstm_h(x)
#        print("Size after lstm horizantal",x.size())
        x = rearrange(x, 'h1 (b w1) (c h2 w2) -> w1 (b h1) (c h2 w2)', b=b, w2=k_w, h2=k_h)
#        print(x.size())
        x, _ = self.lstm_v(x)
#        print("Size after lstm vertical",x.size())
        x = rearrange(x, 'w1 (b h1) (c h2 w2) -> b (c h2 w2) h1 w1', b=b, w2=k_w, h2=k_h)
#        print(x.size())
#        print("============")
        return x
        

class PL_ReNet_LeNet10(LightningModule):
    """Network definition using Pytorch-Lightining and requires to define several funtions (please read the documentation).

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
        parser.add_argument("--optimizer", default="SGD")
        return parser 

    def __init__(self, lr, momentum, optimizer, dataset, **kwargs):
        super(PL_ReNet_LeNet10, self).__init__()
        self.save_hyperparameters(
          'lr',
          'optimizer',
          'momentum',
          'dataset'
        )

        # Image transformation using Kornia
        self.transform_training = nn.Sequential(
#            RandomCrop(size=(32,32), padding=4),
#            RandomResizedCrop(size=(32,32)),
            RandomHorizontalFlip(p=0.25),
            RandomVerticalFlip(p=0.25),
#            RandomAffine(p=0.25, degrees = 0, translate=(1/16,1/16)),
            Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        )
        
        self.transform_eval = nn.Sequential(
#             RandomCrop(size=(32,32), padding=4),  
#             RandomResizedCrop(size=(32,32)),
             Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        )   
        self.dataset = dataset
        self.model = nn.Sequential(
           ReNet(2 * 2 * 3, 160, kernel_size=(2, 2)), 
           nn.Dropout(p=0.3),
           ReNet(2 * 2 * 320, 160, kernel_size=(2, 2)),
           nn.Dropout(p=0.3),
           ReNet(2 * 2 * 320, 160, kernel_size=(2, 2)),
           nn.Dropout(p=0.3),
#           ReNet(2 * 2 * 320, 160, kernel_size=(2, 2)),
#           nn.Dropout(p=0.3),
#           ReNet(2 * 2 * 320, 160, kernel_size=(2, 2)),
#           nn.Dropout(p=0.3),
           nn.Flatten(),
#           nn.Linear(320 * 1 * 1, 4096),
           nn.Linear(320 * 4 * 4, 4096),
           nn.Dropout(p=0.3),
           nn.ReLU(),
           nn.Linear(4096, 10),
        )
        
        self.optimizer = optimizer
        self.epoch = 0
        self.lr = lr
        self.momentum = momentum

        self.metric = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def on_epoch_start(self):
        print('\n')
        
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
        if self.optimizer == 'SGD':
            return optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=5e-4)
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




class PL_ReNet_LeNet(LightningModule):
    """Network definition using Pytorch-Lightining and requires to define several funtions (please read the documentation).

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
        parser.add_argument("--optimizer", default="SGD")
        return parser 

    def __init__(self, lr, momentum, optimizer, dataset, **kwargs):
        super(PL_ReNet_LeNet, self).__init__()
        self.save_hyperparameters(
          'lr',
          'optimizer',
          'momentum',
          'dataset'
        )

        # Image transformation using Kornia
        self.transform_training = nn.Sequential(
#            RandomCrop(size=(32,32), padding=4),
            RandomResizedCrop(size=(64,64)),
            RandomHorizontalFlip(p=0.25),
            RandomVerticalFlip(p=0.25),
            RandomAffine(p=0.25, degrees = 0, translate=(1/16,1/16)),
            Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        )
        
        self.transform_eval = nn.Sequential(
#             RandomCrop(size=(32,32), padding=4),  
             RandomResizedCrop(size=(64,64)),
             Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        )   
        self.dataset = dataset
        self.model = nn.Sequential(
           ReNet(2 * 2 * 3, 160, kernel_size=(2, 2)), 
           nn.Dropout(p=0.3),
           ReNet(2 * 2 * 320, 160, kernel_size=(2, 2)),
           nn.Dropout(p=0.3),
           ReNet(2 * 2 * 320, 160, kernel_size=(2, 2)),
           nn.Dropout(p=0.3),
           ReNet(2 * 2 * 320, 160, kernel_size=(2, 2)),
           nn.Dropout(p=0.3),
           ReNet(2 * 2 * 320, 160, kernel_size=(2, 2)),
           nn.Dropout(p=0.3),
           nn.Flatten(),
#           nn.Linear(320 * 1 * 1, 4096),
           nn.Linear(320 * 2 * 2, 4096),
           nn.Dropout(p=0.3),
           nn.ReLU(),
           nn.Linear(4096, 200),
        )
        
        self.optimizer = optimizer
        self.epoch = 0
        self.lr = lr
        self.momentum = momentum

        self.metric = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def on_epoch_start(self):
        print('\n')
        
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
        if self.optimizer == 'SGD':
            return optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=5e-4)
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

if __name__ == '__main__':

    renet = nn.Sequential(
       ReNet(2 * 2 * 3, 128, kernel_size=(2, 2)), 
       ReNet(2 * 2 * 256, 128, kernel_size=(2, 2)),
       nn.Flatten(),
       nn.Linear(256 * 8 * 8, 4096),
       nn.ReLU(),
       nn.Linear(4096, 10),
    )
    print(renet.summary())
    a = torch.rand((128, 3, 32, 32))
    x = renet(a)
    print(x.size())
    assert x.size() == (128, 10)
    print('Success')
