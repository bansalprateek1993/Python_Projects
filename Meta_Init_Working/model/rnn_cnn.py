import torch
import torch.nn as nn
from torch.nn import functional as func
import torch.optim as optim
import pytorch_lightning as pl

import numpy as np
from argparse import ArgumentParser
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from kornia.augmentation import RandomCrop, Normalize,RandomHorizontalFlip, RandomVerticalFlip,RandomAffine
from einops import rearrange
import math

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
        if self.dataset == "MNIST":
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        elif self.dataset == "CIFAR":
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

class LeNet_updated(nn.Module):
    """Definition of Lenet
    """

    def __init__(self):
        super(LeNet_updated, self).__init__()
        #For CIFAR data input channels are 3
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        return x


class ReNet(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=(2, 2), rnn='GRU', depth=(1, 1)):
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

    def forward(self, x):
        k_w, k_h = self.kernel_size
        b, c, h, w = x.size()
        assert h % k_h == 0 and w % k_w == 0, 'input size does not match with kernel size'
        x = rearrange(x, 'b c (h1 h2) (w1 w2) -> h1 (b w1) (c h2 w2)', w2=k_w, h2=k_h)

        x, _ = self.lstm_h(x)
        x = rearrange(x, 'h1 (b w1) (c h2 w2) -> w1 (b h1) (c h2 w2)', b=b, w2=k_w, h2=k_h)
        x, _ = self.lstm_v(x)
        x = rearrange(x, 'w1 (b h1) (c h2 w2) -> b (c h2 w2) h1 w1', b=b, w2=k_w, h2=k_h)
        return x



class LeReNet(nn.Module):
    """Creating the ReNet(2 RNN layer) and LeNet(2 CNN layer) in parallel.

    Explaination for foward: Dividing the 3*32*32 image into 4 images of size (3*16*16) and passing the 2 images to ReNet and 2 Images to LeNet and
    then concatenating all of them after flattening it to pass it into the fully connected layer. Using 2 FC layer.
    Able to run Metainit with this too.  
    """
    def __init__(self, dataset):
        super(LeReNet, self).__init__()
        self.rnn = nn.Sequential(ReNet(2 * 2 * 3, 128, kernel_size=(2, 2)),
                      ReNet(2 * 2 * 256, 128, kernel_size=(2, 2)))

        self.cnn = LeNet_updated()
        self.flatten = nn.Flatten()
        self.dataset = dataset

        if self.dataset == "CIFAR":
          self.fc1 = nn.Linear(8224, 4096) #256*4*4*2(for 2 rnn blocks) = 8192  and 16*2(for to cnn blocks) 8192 + 32 = 8224
        elif self.dataset == "TImageNet":
          self.fc1 = nn.Linear(33568, 4096) #256 * 8 * 8 * 2(for 2 rnn blocks) = 32768  and 16 * 5 * 5 * 2(for to cnn blocks) 32,768 + 800 = 33,568
        
        if self.dataset == "CIFAR":
          self.fc2 = nn.Linear(4096, 10) 
        elif self.dataset == "TImageNet":
          self.fc2 = nn.Linear(4096, 200)
        
        

    def forward(self, x):
        a,b,c,d = x.size()
        x = rearrange(x, 'a b c d -> a c d b')
#        x = x.reshape((128, 32, 32, 3))
        x = x.cpu().numpy()
        images_resized = np.zeros([0, 4, int(c/2), int(d/2), 3], dtype=np.uint8)
#        print(type(images_resized))
        for image in range(x.shape[0]):
            temp = np.array([x[image][i:i + int(c/2), j:j + int(c/2)] for j in range(0, c, int(c/2)) for i in range(0, c, int(c/2))])
            images_resized = np.append(images_resized, np.expand_dims(temp, axis=0), axis=0)
        
        images_resized = torch.from_numpy(images_resized).cuda()
        a,b,c,d,e = images_resized.size()
        images_resized = rearrange(images_resized, 'a b c d e -> b a e c d')
#        images_resized = images_resized.reshape((4, 128, 3, 16, 16))
#        print(images_resized.shape)
        
        #Passing different images to different networks, 2 (16*16*3) images are passed to RNN
        #and 2 images of (16*16*3) iamges are passed to CNN
        x1 = self.rnn(images_resized[0])
        x2 = self.rnn(images_resized[1])
        x3 = self.cnn(images_resized[2])
        x4 = self.cnn(images_resized[3])
      
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x3 = self.flatten(x3)
        x4 = self.flatten(x4)
        

        concat_tensor = torch.cat((x1, x2, x3, x4), 1)
        
        x = func.relu(self.fc1(concat_tensor))
        x = func.relu(self.fc2(x))
        return x




class PL_LeReNet(LightningModule):
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
        parser.add_argument("--optimizer", default="SGD")
        return parser 
        
    def __init__(self, lr, momentum, optimizer, dataset, **kwargs):
        super(PL_LeReNet, self).__init__()
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
                # RandomHorizontalFlip(p=0.5),
                Normalize(mean=torch.Tensor([0.485]), std=torch.Tensor([0.229])),
            )

            self.transform_eval = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485]), std=torch.Tensor([0.229])),
            )
        elif self.dataset == "CIFAR":
            self.transform_training = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )

            self.transform_eval = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )
        elif self.dataset == "TImageNet":
            self.transform_training = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )

            self.transform_eval = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )
            
        self.rcnn = LeReNet(self.dataset)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum

        self.metric = Accuracy()

    def forward(self, x):
        x = self.rcnn(x)
        return x
    
    def on_epoch_start(self):
        print('\n')


    def training_step(self, batch, batch_idx):
        x, y = batch  # batch, channel, w, h
        x = self.transform_training(x)
        y_hat = self(x)

        accuracy = self.metric(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
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

    renet2 = LeReNet("TImageNet")
    # renet1 = nn.Sequential(
    #     ReNet(2 * 2 * 3, 128, kernel_size=(2, 2)),
    #     ReNet(2 * 2 * 256, 128, kernel_size=(2, 2)),
    #     nn.Flatten(),
    #     nn.Linear(256 * 4 * 4, 4096),
    #     nn.ReLU(),
    #     nn.Linear(4096, 10),
    # )
    renet2 = renet2.cuda()
    print(renet2)
    a = torch.rand((128, 3, 64, 64))
    # x = renet1(a)
    x1 = renet2(a)
    print(x1.size())
    assert x1.size() == (128, 10)
    print('Success')