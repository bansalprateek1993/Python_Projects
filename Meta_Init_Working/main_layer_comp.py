import os
import numpy as np
import torch
import pytorch_lightning as pl
from data.mnist import PL_MNIST
from data.svhn import PL_SVHN
from data.cifar10 import PL_CIFAR
from data.tiny_imagenet import PL_TIMAGENET

from data.mnist_meta import MNIST_META
from data.svhn_meta_rnn_noise import SVHN_META
from data.tiny_iamgenet_meta_rnn_noise import TIMAGENET_META

from data.cifar_meta_rnn_curriculum import CIFAR_META
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

from model.lenet import PL_LeNet
from model.basic import PL_Basic
from model.rnn_cnn import PL_LeReNet
from model.renet_bk import PL_ReNet_LeNet,SomeCallback as SomeCallback1

from model.wide_resnet_without_bn_and_scon import PL_Wide_Resnet,SomeCallback
from argparse import ArgumentParser
from cka import linear_CKA, kernel_CKA
import torch.nn.init as init
from time import time
from torch import nn
from einops import rearrange
import math

def activation_layers(model, a):
    activation_model = []
    if args.layer_comp == 1:
      for i in list([10,8,6]):
        renet_renet1 = nn.Sequential(*list(model.model.children())[:-i])
        for param in renet_renet1.parameters():
          param.requires_grad = False
        renet1_output = renet_renet1(a)
#        print("RENET 1 output is:",renet1_output.size())
        x = np.array(rearrange(renet1_output, 'b c h w -> b h w c'))
#        print("RENET 1 new shape is:",x.shape)
        avg_acts1 = np.mean(x, axis=(1,2))
        print("Appending in:", i)
        activation_model.append(avg_acts1)
        for param in renet_renet1.parameters():
          param.requires_grad = True
    return activation_model

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

'''
only orthogonal initilization
'''
def conv_orthogonal_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
'''
only linear layer initialization after metainit with random initailization
'''
def Linear_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
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
    parser.add_argument("--layer_comp", default=1, type=int)

#    parser = PL_LeNet.add_model_specific_args(parser)
#    parser = PL_Wide_Resnet.add_model_specific_args(parser)
#    parser = PL_Basic.add_model_specific_args(parser)
#    parser = PL_LeNet1.add_model_specific_args(parser)
    parser = PL_ReNet_LeNet.add_model_specific_args(parser)
#    parser = PL_LeReNet.add_model_specific_args(parser)

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
      model = PL_ReNet_LeNet(args.lr, args.momentum, args.optimizer, args.dataset)
      model1 = PL_ReNet_LeNet(args.lr, args.momentum, args.optimizer, args.dataset)
    elif model_dict[1] == "PL_LeReNet":
      model = PL_LeReNet(args.lr, args.momentum, args.optimizer, args.dataset)

        
    print(model)
    print(args.optimizer)
    print(args.dataset)
    print(args.lr)
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
            num_workers=4
        )
    elif args.dataset == "SVHN":
        dm = PL_SVHN(
            batch_size=128,
            data_dir='/netscratch/bansal/dataset/SVHN/origin_data/',
            num_workers=4
        )
        dm1 = SVHN_META(
            model,
            batch_size=32,
            data_dir='/netscratch/bansal/dataset/SVHN/origin_data/',
            num_workers=4
        )
    elif args.dataset == "TImageNet":
        dm = PL_TIMAGENET(
            batch_size=128,
            data_dir='/netscratch/bansal/dataset/tiny-imagenet-200/',
            num_workers=4
        )
        dm1 = TIMAGENET_META(
            model,
            batch_size=32,
            data_dir='/netscratch/bansal/dataset/tiny-imagenet-200/',
            num_workers=4
        )

    torch.save({
        'model_state_dict': model.state_dict(),
    }, "initial_model.pt")

    a = torch.rand((1000, 3, 32, 32))
    activation_wo_metainit = activation_layers(model, a)
    print("Activation without metainit:", activation_wo_metainit[0].shape)


#    print("Applying Xavier Conv weights")
#    model.apply(conv_init)
    
#    print("Applying Delta Orthogonal Conv weights")
#    model.apply(conv_delta_orthogonal_init)
      

    print("Starting Meta init procedure ---> Going to the _meta file")
    ##Updating model parameter with METAINIT###
    dm1()
#    print(list(model.parameters())[0])

#    print("Loading metainit weights of SVHN dataset for CIFAR TRAINING 5 layers")
#    checkpoint = torch.load("model_svhndata_3layer.pt")
#    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.cpu()
    activation_w_metainit = activation_layers(model, a)
    print("Activation without metainit:", activation_w_metainit[0].shape)
    print("Activation without metainit:", activation_w_metainit[1].shape)
    print("Activation without metainit:", activation_w_metainit[2].shape)
            
    for j in range(3):
      print("CKA value for layer:", j)
      print('Linear CKA: {}'.format(linear_CKA(activation_w_metainit[j], activation_wo_metainit[j])))
      print('RBF Kernel CKA: {}'.format(kernel_CKA(activation_w_metainit[j], activation_wo_metainit[j])))
#    print("Only changing the weights of fully connected layers")
#    model.apply(Linear_init)
    
    
    mlf_logger = MLFlowLogger(experiment_name=args.exp_name,
                              tracking_uri="file:" + args.output_folder)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=2,
        verbose=True,
        monitor='val_acc'
    )
    
#    trainer_metaInit = pl.Trainer(gradient_clip_val=1.0, callbacks=[SomeCallback(model)], max_epochs=200, gpus=1)
    trainer_metaInit = pl.Trainer(max_epochs=150, callbacks=[SomeCallback1(model),checkpoint_callback], logger=mlf_logger,gpus=1)
    print("Training started") 
    time0 = time()
#    trainer_metaInit.fit(model, dm)
    
#    trainer_metaInit.test()
#    print("Best Validation accuracy is:", model.best_acc)
    print("\nTraining Time for Metainit(in minutes) =", (time() - time0) / 60)
