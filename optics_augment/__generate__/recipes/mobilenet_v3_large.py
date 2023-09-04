# Patrick Müller (c) 2022, 09.12.2022
import os

import torch
import torch.optim as optim # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from . import _basic_training


def get_training_setup_kuan_wang(model_dnn,**kwargs):
    """!"""  # note also: https://github.com/fadel/pytorch_ema and https://github.com/lukemelas/EfficientNet-PyTorch/issues/66
    # https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet
    batch_size = 128 #256
    learning_rate = 0.05
    momentum = 0.9
    weight_decay = 4e-5 # l2 weight decay for RMSprop, https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    num_epochs = 150

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model_dnn.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, \
        momentum=momentum, weight_decay=weight_decay, centered=False, foreach=None)

    #cosine_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,)
    #cosine_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=- 1, verbose=False)
    cosine_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,\
        steps_per_epoch=2,anneal_strategy='cos',epochs=num_epochs) # steps_per_epoch = 2 ('train','val')

    training_setup = {}
    training_setup["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    training_setup["num_epochs"] = kwargs.get("num_epochs",num_epochs)
    training_setup["batch_size"] = kwargs.get("batch_size",batch_size)
    training_setup["criterion"] = criterion
    training_setup["optimizer"] = optimizer
    training_setup["scheduler"] = cosine_lr_scheduler

    return training_setup


def get_training_setup_kuan_wang_warm_restarts(model_dnn,**kwargs):
    """!"""
    batch_size = 128 #256
    learning_rate = 0.05
    momentum = 0.9
    weight_decay = 4e-5 # l2 weight decay for RMSprop, https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    num_epochs = 150
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.RMSprop(model_dnn.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, \
    #    momentum=momentum, weight_decay=weight_decay, centered=False, foreach=None)
    optimizer = optim.SGD(model_dnn.parameters(), lr=learning_rate, \
        momentum=momentum, weight_decay=weight_decay)
    # currently used: RMSprop ... 18:41, scheduled this optimizer for further exploration

    #cosine_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,)
    cosine_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1,\
        T_mult=1, eta_min=0, last_epoch=- 1, verbose=False)

    training_setup = {}
    training_setup["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    training_setup["num_epochs"] = kwargs.get("num_epochs",num_epochs)
    training_setup["batch_size"] = kwargs.get("batch_size",batch_size)
    training_setup["criterion"] = criterion
    training_setup["optimizer"] = optimizer
    training_setup["scheduler"] = cosine_lr_scheduler

    return training_setup


def load_recipe(model_dnn,setup=None,take_params = "kuan_wang_warm_restarts",**kwargs):
    """Demo:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Preparations"""

    if take_params == "kuan_wang":
        setup["training"] = get_training_setup_kuan_wang(model_dnn,**kwargs)
        name = model_dnn.__name__ + "_kuan_wang"
    elif take_params == "kuan_wang_warm_restarts":
        setup["training"] = get_training_setup_kuan_wang_warm_restarts(model_dnn,**kwargs)
        name = model_dnn.__name__ + "_kuan_wang_warm_restarts"
    else:
        setup["training"] = _basic_training.get_training_setup(model_dnn,**kwargs)
        name = model_dnn.__name__ + "_pt_vision"

    return setup,model_dnn,name
