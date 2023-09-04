# Patrick Müller (c) 2023  (classification models pytorch)
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms 


def get_training_setup(model_dnn,mode="sgd",**kwargs):
    """!
    Default values as from:
    https://github.com/pytorch/vision/tree/main/references/classification
    https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
    --> baseline models
    """
    batch_size = 32
    learning_rate = 0.1 # 8x V100 GPUs 
    momentum =0.9
    step_size = 30 # learning rate decay (lr_scheduler)
    gamma= 0.1 #  learning rate decay rate (lr_scheduler)
    weight_decay = 1e-4 # l2 weight decay
    num_epochs = 90

    criterion = nn.CrossEntropyLoss()

    if mode == "rmsprop":
        optimizer = optim.RMSprop(model_dnn.parameters(), lr=learning_rate, #alpha=0.99, eps=1e-08, \
            momentum=momentum, weight_decay=weight_decay, centered=False, foreach=None)
    elif mode == "sgd":
        optimizer = optim.SGD(model_dnn.parameters(), lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    # Decay LR by a factor of gamma every step_size epochs
    selected_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    training_setup = {}
    training_setup["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    training_setup["num_epochs"] = kwargs.get("num_epochs",num_epochs)
    training_setup["batch_size"] = kwargs.get("batch_size",batch_size)
    training_setup["criterion"] = criterion
    training_setup["optimizer"] = optimizer
    training_setup["scheduler"] = selected_lr_scheduler

    return training_setup
