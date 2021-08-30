import os
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.init as init
from data.mnist import PL_MNIST
from data.svhn import PL_SVHN
from data.cifar10 import PL_CIFAR
from data.mnist_meta import MNIST_META
from data.svhn_meta import SVHN_META
from data.cifar_meta import CIFAR_META
from model.lenet import PL_LeNet
from model.wide_resnet_without_bn_and_scon import PL_Wide_Resnet
import matplotlib.pyplot as plt
from model.basic import PL_Basic
from argparse import ArgumentParser
from time import time
import pandas as pd
from torch import nn
import math

'''
Commenting to add for wide_Resnet
def weights_init_normal(m):
    #Takes in a module and initializes all linear layers with weight
    #   values taken from a normal distribution

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        #        m.weight.data.normal_(0.0,1/np.sqrt(y))
        m.weight.data.normal_(0.0, 0.1)
        # m.bias.data should be 0
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        # m.weight.data shoud be taken from a normal distribution
        #        m.weight.data.normal_(0.0,1/np.sqrt(y))
        #        torch.nn.init.xavier_uniform_(m.weight)
        #        m.weight.data.normal_(0.0,0.1)
        m.weight.data.normal_(0.0, 0.1)
        # m.bias.data should be 0
        m.bias.data.fill_(0)
        
        
def weights_init_xavier(m):
    #Takes in a module and initializes all linear layers with weight
    #  values taken from a normal distribution

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        #        m.weight.data.normal_(0.0,1/np.sqrt(y))
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data should be 0
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        # m.weight.data shoud be taken from a normal distribution
        #        m.weight.data.normal_(0.0,1/np.sqrt(y))
        torch.nn.init.xavier_uniform_(m.weight)
        #        m.weight.data.normal_(0.0,0.1)
        #m.weight.data.normal_(0.0, 0.01)
        # m.bias.data should be 0
        m.bias.data.fill_(0)
'''
def conv_xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        
def conv_guas_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.1)
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
'''
conv delta orhogonalization deprecated
'''
def conv_delta2_orthogonal_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        conv_delta_orthogonal_(m.weight, gain=np.sqrt(2))
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
    # Dataset
    parser = ArgumentParser()

    # Dataset parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", default='MNIST')
    parser.add_argument("--num_workers", default=1, type=int)

    parser.add_argument('--model', default='LeNet:PL_LeNet', type=str)

    #    parser = PL_LeNet.add_model_specific_args(parser)
    parser = PL_Wide_Resnet.add_model_specific_args(parser)
    #    parser = PL_Basic.add_model_specific_args(parser)
    #    parser = PL_LeNet1.add_model_specific_args(parser)

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
            batch_size=60,
            data_dir='/netscratch/bansal/dataset/MNIST/origin_data/',
            num_workers=1
        )
    elif args.dataset == "CIFAR":
        dm = PL_CIFAR(
            batch_size=60,
            data_dir='/netscratch/bansal/dataset/CIFAR/origin_data/',
            num_workers=4
        )
        dm1 = CIFAR_META(
            model,
            batch_size=60,
            data_dir='/netscratch/bansal/dataset/CIFAR/origin_data/',
            num_workers=1
        )
    elif args.dataset == "SVHN":
        dm = PL_SVHN(
            batch_size=60,
            data_dir='/netscratch/bansal/dataset/SVHN/origin_data/',
            num_workers=4
        )
        dm1 = SVHN_META(
            model,
            batch_size=60,
            data_dir='/netscratch/bansal/dataset/SVHN/origin_data/',
            num_workers=1
        )

    torch.save({
        'model_state_dict': model.state_dict(),
    }, "initial_model.pt")

#    print("Calculating norm of initial weights ======")
    print("Applying Gaus weights ======")
    model.apply(conv_guas_init)

    norm_normal = []
    params = [p for p in model.parameters()
              if p.requires_grad and len(p.size()) >= 2]

    for j, (p) in enumerate(params):
        norm = p.data.norm().item()
        norm_normal.append(norm)

    print(norm_normal)

#    print("APPLYING XAVIER CONV WEIGHTS===============")
#    model.apply(conv_init)

    print("APPLYIGN DELTA WEIGHTS")
    model.apply(conv_delta2_orthogonal_init)
    norm_delta = []
    params = [p for p in model.parameters()
              if p.requires_grad and len(p.size()) >= 2]

    for j, (p) in enumerate(params):
        norm = p.data.norm().item()
        norm_delta.append(norm)

    print(norm_delta)

    print("APPLYIGN XAVIER WEIGHTS")
    model.apply(conv_xavier_init)
    norm_xavier = []
    params = [p for p in model.parameters()
              if p.requires_grad and len(p.size()) >= 2]

    for j, (p) in enumerate(params):
        norm = p.data.norm().item()
        norm_xavier.append(norm)

    print(norm_xavier)
    
    print("Applying meta init weights")
    ##Updating model parameter with METAINIT###
##    dm1()
    checkpoint1 = torch.load("model.pt")
    print("Loading MetaInit weights")
    model.load_state_dict(checkpoint1['model_state_dict'])

    norm_meta = []
    for j, (p) in enumerate(params):
        norm = p.data.norm().item()
        norm_meta.append(norm)

    print(norm_meta)
    
    layer_number = []
    for j, (p) in enumerate(params):
        layer_number.append(j)

    evaluation_dict = {"layer_number" : layer_number,
                       "norm_normal_0.1" : norm_normal,
                       "norm_delta" : norm_delta,
                       "norm_xavier" : norm_xavier,
                       "norm_meta" : norm_meta}
                       
    evaluation_dict_df = pd.DataFrame.from_dict(evaluation_dict, orient='columns')
    plt.plot( 'layer_number', 'norm_normal_0.1', data = evaluation_dict_df, marker='|', markerfacecolor='blue', markersize=2, color='skyblue', linewidth=1)
    plt.plot( 'layer_number', 'norm_delta', data = evaluation_dict_df, marker='|', markerfacecolor='green', markersize=2, color='green', linewidth=1)
    plt.plot( 'layer_number', 'norm_xavier', data = evaluation_dict_df, marker='|', markerfacecolor='red', markersize=2, color='red', linewidth=1)
    plt.plot( 'layer_number', 'norm_meta', data = evaluation_dict_df, marker='o', markerfacecolor='black', markersize=2, color='black', linewidth=1)

    plt.legend()
    plt.savefig('Weights matrix1.png')
    plt.close()
#    print(list(model.parameters())[0])

    #    print("Udpated model Parameters :", list(model.parameters()))
    # trainer_metaInit = pl.Trainer(max_epochs=30, gpus=1)
    # print("Training started")
    # time0 = time()
    # trainer_metaInit.fit(model, dm)
    # print("\nTraining Time for Metainit(in minutes) =", (time() - time0) / 60)


    # plt.plot(list(layers), list(norms_initial))
    # plt.plot(list(layers), list(norms_after_metainit))
    # plt.plot(list(layers), list(norms_xavier))
    #
    # plt.title('Comparision of weight norm with different layer')
    # plt.ylabel('Weight Norm')
    # plt.xlabel('Number of Layer')
    # plt.legend(['initial_Gaussian', 'metainit','xavier'], loc='bottom right')
    # # plt.savefig('Different_nc_values_for_different_architecture.png')
    # plt.show()
    # plt.clf()