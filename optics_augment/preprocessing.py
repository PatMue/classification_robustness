# ImageNet-1k, 100,  Patrick Müller 2022 (modified from as specified)
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from augmix import AugMixDataset


class DefaultNormalize:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    @staticmethod
    def get():
        return transforms.Normalize(mean=DefaultNormalize.mean,std=DefaultNormalize.std)


def imagenet_normalization():
    preprocess = transforms.Compose([
        transforms.Normalize(mean=DefaultNormalize.mean,std=DefaultNormalize.std)])
    return preprocess 


def opticsaugment(setup):

    preprocess = imagenet_normalization()

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            preprocess
        ]),
    }

    traindir = os.path.join(setup["root_dir"], 'train')
    valdir = os.path.join(setup["root_dir"], 'val')
    num_workers = setup["training"]["num_workers"]
    batch_size = setup["training"]["batch_size"]
	
    train_dataset = datasets.ImageFolder(traindir, data_transforms['train'])
    val_dataset = datasets.ImageFolder(valdir,data_transforms['val'])
    image_datasets = {'train':train_dataset,'val':val_dataset}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],\
                                batch_size=batch_size,
                                shuffle=True if x=='train' else False, 
                                pin_memory=torch.cuda.is_available(),
                                num_workers=num_workers)
                  for x in ['train', 'val']}

    return image_datasets,dataloaders



def augmix(setup):
    """!"""	

    normalize = transforms.Compose([\
		transforms.Normalize(mean=DefaultNormalize.mean,std=DefaultNormalize.std)
		])

    preprocess = transforms.Compose([transforms.ToTensor(),normalize])


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            preprocess
        ]),
    }

    traindir = os.path.join(setup["root_dir"], 'train')
    valdir = os.path.join(setup["root_dir"], 'val')
    num_workers = setup["training"]["num_workers"]
    batch_size = setup["training"]["batch_size"]
	
    train_dataset = datasets.ImageFolder(traindir, data_transforms['train'])
    
    pipelined = setup["training"].get("pipelined",False)
    if pipelined:
        train_dataset = AugMixDataset(train_dataset, transforms.ToTensor(),pipelined=True)	
    else:
        train_dataset = AugMixDataset(train_dataset, preprocess,pipelined=False)	

    val_dataset = datasets.ImageFolder(valdir,data_transforms['val'])
    image_datasets = {'train':train_dataset,'val':val_dataset}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],\
                                batch_size=batch_size,
                                shuffle=True if x=='train' else False, 
                                pin_memory=torch.cuda.is_available(),
                                num_workers=num_workers)
                  for x in ['train', 'val']}

    return image_datasets,dataloaders


def basic(setup,**kwargs): # shuffle=True for val before 10.02.2023
    """!"""
    # Data augmentation and normalization for training
    # Just normalization for validation
    # just as suggested by pytorch etc. 

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), # mirroring
            transforms.ToTensor(),
            transforms.Normalize(mean=DefaultNormalize.mean,std=DefaultNormalize.std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=DefaultNormalize.mean,std=DefaultNormalize.std)
        ]),
    }

    num_workers = setup["training"]["num_workers"]
    batch_size = setup["training"]["batch_size"]


    image_datasets = {x: datasets.ImageFolder(os.path.join(setup["root_dir"], x),
                        data_transforms[x])  for x in ['train', 'val']
                      }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], \
                        batch_size=batch_size,
                        shuffle=True if x=='train' else False, \
                        pin_memory=torch.cuda.is_available(),\
                        num_workers=num_workers) \
                    for x in ['train', 'val']
			       }

    return image_datasets,dataloaders


def as_from_kuan_wang(setup):
    raise NotImplementedError("Deprecated and no longer available. Replace with basic()")
    """!
    modified from https://github.com/kuan-wang/pytorch-mobilenet-v3
    """
