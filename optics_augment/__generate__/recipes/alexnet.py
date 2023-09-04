# Patrick Müller (c) 2023, 09.12.2022
import os

import torch
import torch.optim as optim # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from . import _basic_training


def load_recipe(model_dnn,setup=None,**kwargs):
    """
    https://github.com/pytorch/vision/tree/main/references/classification#Alexnet
    """
    training_setup = {}
    training_setup["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    training_setup["num_epochs"] = kwargs.get("num_epochs",90 + 10)
    training_setup["batch_size"] = kwargs.get("batch_size",950) # to use 11.1GB/12GB
    training_setup["criterion"] = nn.CrossEntropyLoss()
    training_setup["optimizer"] = optim.SGD(model_dnn.parameters(), lr=1e-2, momentum=0.9,weight_decay=1e-4)
    training_setup["scheduler"] = lr_scheduler.StepLR(training_setup["optimizer"], step_size=30, gamma=0.1)

    setup["training"] = training_setup 

    name = model_dnn.__name__ + "_sgd"
    
    return setup,model_dnn,name
