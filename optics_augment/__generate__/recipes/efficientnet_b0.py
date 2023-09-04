# Patrick Müller (c) 2022, 09.12.2022
import os

import torch
import torch.optim as optim # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from . import _basic_training


def get_training_setup(model_dnn,mode="rmsprop",**kwargs):
    """!
    https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet
    """
    batch_size = 100 #256
    learning_rate = 0.08
    momentum = 0.9
    weight_decay = 1e-5 # l2 weight decay for RMSprop, https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    num_epochs = 200
    criterion = nn.CrossEntropyLoss()

    if mode.lower() == "rmsprop":
        optimizer = optim.RMSprop(model_dnn.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, \
        momentum=momentum, weight_decay=weight_decay, centered=False, foreach=None)
        # currently used: RMSprop ... 18:41, scheduled this optimizer for further exploration
        #lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,)
    elif mode.lower() == "sgd":
        optimizer = optim.SGD(model_dnn.parameters(), lr=learning_rate, \
           momentum=momentum, weight_decay=weight_decay)

    selected_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,1,#3,#16,\
        T_mult=1, eta_min=0, last_epoch=- 1, verbose=False)

    training_setup = {}
    training_setup["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    training_setup["num_epochs"] = kwargs.get("num_epochs",num_epochs)
    training_setup["batch_size"] = kwargs.get("batch_size",batch_size)
    training_setup["criterion"] = criterion
    training_setup["optimizer"] = optimizer
    training_setup["scheduler"] = selected_lr_scheduler

    return training_setup


def load_recipe(model_dnn,setup=None,**kwargs):
    """Demo:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Preparations"""
    setup["training"] = get_training_setup(model_dnn,mode="sgd",**kwargs)
    #savename = "efficientnet_b0_rmsprop.pt"
    name = model_dnn.__name__ + "_sgd"
    
    return setup,model_dnn,name
