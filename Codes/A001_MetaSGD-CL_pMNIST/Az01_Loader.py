###===###
# Coder:        Nic K
# Affiliation:  The Australian National University
#===
# Code purpose:
#   Supporting script, prepares data loader

import  random
import  math
import  numpy                   as      np

import  torch
import  torch.nn                as      nn
import  torch.optim             as      optim
import  torch.nn.functional     as      F

import  torchvision
import  torchvision.transforms  as      transforms
from    torchvision             import  datasets, transforms

###===###
def LoaderAz01(My_BS):
    kwargs = {'pin_memory': True}

    train_loader = \
        torch.utils.data.DataLoader(
            datasets.MNIST(
                './data',
                train = True, download = True,
                transform = \
                    transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size = My_BS,
            shuffle = True,
            **kwargs)     

    test_loader = \
        torch.utils.data.DataLoader(
            datasets.MNIST(
                './data',
                train = False, download = True,
                transform = \
                    transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size = 256,
            shuffle = True,
            **kwargs)

    return train_loader, test_loader
