# Patrick Müller (c) 2022, 09.12.2022
import os

import torch
import torch.optim as optim # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from . import _basic_training


def load_recipe(model_dnn,setup=None,**kwargs):
    """
    https://github.com/pytorch/vision/tree/main/references/classification#Resnet
    """
    
    setup["training"] = _basic_training.get_training_setup(model_dnn,mode="sgd")
    setup["training"]["batch_size"] = kwargs.get("batch_size",70) # to use 11.4GB/12GB  # reduced for new implementation
    setup["training"]["num_epochs"] = kwargs.get("num_epochs",90 + 10)
    setup["training"]["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    
    name = model_dnn.__name__ + "_sgd"

    return setup,model_dnn,name
