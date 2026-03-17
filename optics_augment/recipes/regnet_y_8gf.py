# Patrick Müller (c) 2022-2023, 04.03.2023
import os

import torch
import torch.optim as optim # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from . import _basic_training


def get_training_setup(model_dnn,**kwargs):
    """!
    https://github.com/pytorch/vision/tree/v0.11.3/references/classification#medium-models
    """
    batch_size = kwargs.get("batch_size",64)
    num_epochs = kwargs.get("num_epochs",150)
    learning_rate = 0.8
    momentum = 0.9
    weight_decay = 0.00005 # l2 weight decay
    lr_warmup_epochs = 5
    lr_warmup_decay = 0.1
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model_dnn.parameters(), lr=learning_rate, \
        momentum=momentum, weight_decay=weight_decay)

    cosine_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs - lr_warmup_epochs)

    warmup_lr_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=lr_warmup_decay,
                                                                    total_iters=lr_warmup_epochs)

    sequential_lr_scheduler = lr_scheduler.SequentialLR(
            optimizer,schedulers=[warmup_lr_scheduler, cosine_lr_scheduler],
            milestones=[lr_warmup_epochs]
            )

    training_setup = {}
    training_setup["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    training_setup["num_epochs"] = num_epochs
    training_setup["batch_size"] = batch_size
    training_setup["criterion"] = criterion
    training_setup["optimizer"] = optimizer
    training_setup["scheduler"] = sequential_lr_scheduler

    return training_setup


def load_recipe(model_dnn,setup=None,**kwargs):
    """
    https://github.com/pytorch/vision/tree/v0.11.3/references/classification#medium-models
    """
    setup["training"] = get_training_setup(model_dnn,**kwargs)
    name = model_dnn.__name__ + "_sgd_sequential_lr"

    return setup,model_dnn,name
