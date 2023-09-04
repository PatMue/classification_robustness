# Patrick Müller (c) 2022-2023, 04.03.2023
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from torchvision import models

from . import _basic_training
from . import _preprocessing

from ._augment import OpticsAugment


def get_training_setup(model_dnn,**kwargs):
    """!
    https://github.com/pytorch/vision/tree/v0.11.3/references/classification#medium-models
    """
    batch_size = kwargs.get("batch_size",128)
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

    training_setup = {}
    training_setup["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    training_setup["num_epochs"] = num_epochs
    training_setup["batch_size"] = batch_size
    training_setup["criterion"] = criterion
    training_setup["optimizer"] = optimizer
    training_setup["scheduler"] = cosine_lr_scheduler
    training_setup["warmup_scheduler"] = warmup_lr_scheduler

    return training_setup


def load_recipe(setup=None,model_path=None,device=None,augmix=False,optics_augment=False,**kwargs):
    """
    https://github.com/pytorch/vision/tree/v0.11.3/references/classification#medium-models
    """
    
    raise NotImplementedError
    name="convnext_base"
    model_dnn = models.__dict__[name](pretrained=False)
    model_dnn.train()
    model_dnn = model_dnn.to(device)

    setup["training"] = get_training_setup(model_dnn,**kwargs)
    savename = name + "{}{}.pt"

    setup["training"]["optics_augment"] = OpticsAugment() if optics_augment else None

    preprocessing_fcn = _preprocessing.augmix if augmix else _preprocessing.basic
    image_datasets,dataloaders = preprocessing_fcn(setup)

    savename = savename.format('_augmix' if augmix else '','_bluraugment' if optics_augment else '')
    model_path = os.path.join(setup["model_dir"],f"{os.path.splitext(savename)[0]}_temp.pt")

    setup["model_name"] = os.path.splitext(savename)[0]
    setup["savepath"] = os.path.join(setup["model_dir"],savename)
    setup["model_path"] = model_path

    return setup,model_dnn,image_datasets,dataloaders
