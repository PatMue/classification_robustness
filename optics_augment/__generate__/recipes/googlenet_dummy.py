# Patrick Müller (c) 2022, 09.12.2022

"""
Our networks were trained using the DistBelief [4] distributed machine learning system using modest amount of model and data-parallelism. Although we used CPU based implementation only, a rough estimate suggests that the GoogLeNet network could be trained to convergence using few high-end GPUs within a week, the main limitation being the memory usage. Our training used asynchronous stochastic gradient descent with 0.9 momentum [17], fixed learning rate schedule (decreasing the learning rate by 4% every 8 epochs). Polyak averaging [13] was used to create the final model used at inference time.
(Szegedy et al. 2014, going deeper with convolutions)
"""

import os

import torch
import torch.optim as optim # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms  # https://vitalflux.com/pytorch-load-predict-pretrained-resnet-model/
from torchvision import models # pytorch vision model_zoo (can also set pretrained = False for training)

from . import _basic_training
from . import _preprocessing

from ._augment import BlurAugment


def load_recipe(setup=None,device=None,model_path=None,take_params = "",**kwargs):
    """Demo:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Preparations"""
    model_dnn = models.googlenet(pretrained=False)
    model_dnn.train()
    model_dnn = model_dnn.to(device)
    
    raise NotImplementedError("Currently not implemented")

    #savename = "efficientnet_b0_rmsprop.pt"
    savename = "googlenet_sgd.pt"
    setup["training"] = get_training_setup(model_dnn,mode="")
    image_datasets,dataloaders = _preprocessing.basic(setup)
    model_path = os.path.join(setup["model_dir"],f"{os.path.splitext(savename)[0]}_temp.pt")

    setup["model_name"] = os.path.splitext(savename)[0]
    setup["savepath"] = os.path.join(setup["model_dir"],savename)
    setup["model_path"] = model_path

    #return setup,model_dnn,image_datasets,dataloaders
    return None
