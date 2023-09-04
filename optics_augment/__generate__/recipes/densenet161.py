# Patrick Müller (c) 2022, Update 05.01.2023
import os

import torch
import torch.optim as optim # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from . import _basic_training


def get_training_setup_from_paper(model_dnn,**kwargs):
    """ 
    recipe as in Denseley Connected Conv. Networks - Huang et al. 2018 CVPR for ImageNet
    !"""
    batch_size = 80
    learning_rate = 0.1
    step_size = 30 # learning rate decay (lr_scheduler)
    gamma= 0.1 #  learning rate decay rate (lr_scheduler) --> decay by 10 times every <step_size> epochs
    num_epochs = 90
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_dnn.parameters(), lr=learning_rate)#, momentum=momentum,weight_decay=weight_decay)
    # Decay LR by a factor of gamma every step_size epochs:
    selected_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    training_setup = {}
    training_setup["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    training_setup["num_epochs"] = kwargs.get("num_epochs",num_epochs)
    training_setup["batch_size"] = kwargs.get("batch_size",batch_size)
    training_setup["criterion"] = criterion
    training_setup["optimizer"] = optimizer
    training_setup["scheduler"] = selected_lr_scheduler
    return training_setup
    
    
def load_recipe(model_dnn,setup=None,**kwargs):
    """
    https://github.com/pytorch/vision/tree/main/references/classification#
    https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet161.html#torchvision.models.densenet161
    https://github.com/pytorch/vision/pull/116  (recipe)
    https://github.com/liuzhuang13/DenseNet
    as from the paper
    """
    setup["training"] = get_training_setup_from_paper(model_dnn,**kwargs)        
    name = model_dnn.__name__ + "_sgd"

    return setup,model_dnn,name
