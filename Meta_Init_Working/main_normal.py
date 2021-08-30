import os
import numpy as np
import torch
import pytorch_lightning as pl
from data.mnist import PL_MNIST
from data.svhn import PL_SVHN
from data.cifar10 import PL_CIFAR
from data.tiny_imagenet import PL_TIMAGENET
from data.imagenette import PL_IMAGENETTE

from data.mnist_meta import MNIST_META
from data.svhn_meta_rnn_noise import SVHN_META
from data.tiny_iamgenet_meta_rnn_data import TIMAGENET_META
from data.imagenette_meta_rnn_data import IMAGENETTE_META

from data.cifar_meta_rnn_noise import CIFAR_META
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback



from model.lenet import PL_LeNet
from model.basic import PL_Basic
from model.renet_1_2_layer import PL_ReNet_LeNet10,SomeCallback as SomeCallback1
import torch.nn.init as init
#from model.wide_resnet_lightning import PL_Wide_Resnet,SomeCallback
#from model.wide_resnet_without_scon_paper_changes import PL_Wide_Resnet,SomeCallback
from model.wide_resnet_modified import PL_Wide_Resnet,SomeCallback
from pytorch_lightning.callbacks import Callback
from argparse import ArgumentParser
from time import time
from torch import nn
import math

'''
## OLD INIT MODULE         
## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    #Takes in a module and initializes all linear layers with weight
    #values taken from a normal distribution

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
#        m.weight.data.normal_(0.0,0.1)
    # m.bias.data should be 0
        m.bias.data.fill_(0)
#    elif classname.find('Conv') != -1:
    # m.weight.data shoud be taken from a normal distribution
#        m.weight.data.normal_(0.0,1/np.sqrt(y))
#        torch.nn.init.xavier_uniform_(m.weight)
#        m.weight.data.normal_(0.0,0.1)
#        m.weight.data.normal_(0.0,0.1)
    # m.bias.data should be 0
#        m.bias.data.fill_(0)
'''

'''
xavier weight initilization
'''
def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

'''
Orthogonal weight initialization
'''
def conv_orthogonal_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight)
        init.constant_(m.bias, 0)
'''
conv delta orhogonalization deprecated
'''
def conv_delta_orthogonal_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        conv_delta_orthogonal_(m.weight)
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        
def conv_delta_orthogonal_(tensor, gain=1.):
    r"""Initializer that generates a delta orthogonal kernel for ConvNets.
    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero. See
    algorithm 2 in [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`3 \leq n \leq 5`
        gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
    Examples:
        >>> w = torch.empty(5, 4, 3, 3)
        >>> nn.init.conv_delta_orthogonal_(w)
    """
    if tensor.ndimension() < 3 or tensor.ndimension() > 5:
      raise ValueError("The tensor to initialize must be at least "
                       "three-dimensional and at most five-dimensional")
    
    if tensor.size(1) > tensor.size(0):
      raise ValueError("In_channels cannot be greater than out_channels.")
    
    # Generate a random matrix
    a = tensor.new(tensor.size(0), tensor.size(0)).normal_(0, 1)
    # Compute the qr factorization
    q, r = torch.qr(a)
    # Make Q uniform
    d = torch.diag(r, 0)
    q *= d.sign()
    q = q[:, :tensor.size(1)]
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndimension() == 3:
            tensor[:, :, (tensor.size(2)-1)//2] = q
        elif tensor.ndimension() == 4:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2] = q
        else:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2, (tensor.size(4)-1)//2] = q
        tensor.mul_(math.sqrt(gain))
    return tensor
    
    
if __name__ == '__main__':
    #Dataset
    parser = ArgumentParser()

    # Dataset parameters
    parser.add_argument("--output_folder", default="/netscratch/bansal/dataset/CIFAR/output_folder/demo/")
    parser.add_argument("--exp_name", default="demo")

    # Dataset parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", default='MNIST')
    parser.add_argument("--num_workers", default=1, type=int)

    parser.add_argument('--model', default='LeNet:PL_LeNet', type=str)

#    parser = PL_LeNet.add_model_specific_args(parser)
#    parser = PL_Wide_Resnet.add_model_specific_args(parser)
#    parser = PL_Basic.add_model_specific_args(parser)
#    parser = PL_LeNet1.add_model_specific_args(parser)
    parser = PL_ReNet_LeNet10.add_model_specific_args(parser)
    
    args = parser.parse_args()
    my_dict = args.model
    model_dict = my_dict.split(":")
    criterion = nn.NLLLoss()

    if model_dict[1] == "PL_LeNet":
        model = PL_LeNet(args.lr, args.momentum, args.optimizer, args.dataset)
    elif model_dict[1] == "PL_Basic":
        model = PL_Basic(args.lr, args.momentum, args.optimizer, args.dataset)
    elif model_dict[1] == "PL_Wide_Resnet":
        model = PL_Wide_Resnet(args.lr, args.momentum, args.optimizer, args.dataset, args.depth, args.width, args.dropout)
    elif model_dict[1] == "PL_ReNet_LeNet":
        model = PL_ReNet_LeNet10(args.lr, args.momentum, args.optimizer, args.dataset)


  
    print(args.optimizer)
    print(args.dataset)
    print(model_dict[0])
    print(model_dict[1])
#    print("Initial model Parameters :", list(model.parameters()))
    if args.dataset == "MNIST":
        dm = PL_MNIST(
            batch_size=60,
            data_dir='/netscratch/bansal/dataset/MNIST/origin_data/',
            num_workers=4
        )
        dm1 = MNIST_META(
            model,
            batch_size=32,
            data_dir='/netscratch/bansal/dataset/MNIST/origin_data/',
            num_workers=1
        )
    elif args.dataset == "CIFAR":
        dm = PL_CIFAR(
            batch_size=128,
            data_dir='/netscratch/bansal/dataset/CIFAR/origin_data/',
            num_workers=4
        )
        dm1 = CIFAR_META(
            model,
            batch_size=128,
            data_dir='/netscratch/bansal/dataset/CIFAR/origin_data/',
            num_workers=1
        )
    elif args.dataset == "SVHN":
        dm = PL_SVHN(
            batch_size=32,
            data_dir='/netscratch/bansal/dataset/SVHN/origin_data/',
            num_workers=4
        )
        dm1 = SVHN_META(
            model,
            batch_size=32,
            data_dir='/netscratch/bansal/dataset/SVHN/origin_data/',
            num_workers=1
        )
    elif args.dataset == "TImageNet":
        dm = PL_TIMAGENET(
            batch_size=128,
            data_dir='/netscratch/bansal/dataset/tiny-imagenet-200/',
            num_workers=4
        )
        dm1 = TIMAGENET_META(
            model,
            batch_size=128,
            data_dir='/netscratch/bansal/dataset/tiny-imagenet-200/',
            num_workers=4
        )
    elif args.dataset == "ImageNette":
        dm = PL_IMAGENETTE(
            batch_size=128,
            data_dir='/netscratch/raue/dataset/imagenette2-320/',
            num_workers=4
        )
        dm1 = IMAGENETTE_META(
            model,
            batch_size=100,
            data_dir='/netscratch/raue/dataset/imagenette2-320',
            num_workers=4
        )
        
    
#    print("Before Gaussian:", list(model.parameters())[0])
#    model.apply(weights_init_normal)
#    print("After Gaussian:", list(model.parameters())[0])

#    print("APPLYING CONV WEIGHTS===============")
#    model.apply(conv_init)
    
#    print("APPLYING ORTHOGONAL WEIGHTS INITIALIZATION===============")
#    model.apply(conv_orthogonal_init)
#    model.apply(conv_init)    
    ##Adding gradient clipping as done is metainit paper for plain Networks
#    trainer_normal = pl.Trainer(gradient_clip_val=1.0, callbacks=[SomeCallback(model)], max_epochs=200, gpus=1)

#    print("Loading metainit weights")
#    checkpoint = torch.load("model.pt")
#    model.load_state_dict(checkpoint['model_state_dict'])


#    print("Starting Meta init procedure --->> Going to the _meta file")
    ##Updating model parameter with METAINIT###
#    dm1()
    mlf_logger = MLFlowLogger(experiment_name=args.exp_name,
                          tracking_uri="file:" + args.output_folder)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=2,
        verbose=True,
        monitor='val_acc'
    )
    
#    trainer_normal = pl.Trainer(callbacks=[SomeCallback(model)], max_epochs=200, gpus=1)
    trainer_normal = pl.Trainer(gradient_clip_val=4.0,max_epochs=150, callbacks=[SomeCallback1(model),checkpoint_callback], logger=mlf_logger,gpus=1)
    time0 = time()
    trainer_normal.fit(model, dm)
    trainer_normal.test()
    print("\nTraining Time without Metainit(in minutes) =", (time() - time0) / 60)
