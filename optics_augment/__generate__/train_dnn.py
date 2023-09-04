"""

#################################
    (c) Patrick Müller 2023 
#################################

If you find this useful in your research, please cite:

@article{mueller2023_opticsbench,
	author   = {Patrick Müller, Alexander Braun and Margret Keuper},
	title    = {Classification robustness to common optical aberrations},
	journal  = {Proceedings of the International Conference on Computer Vision Workshops (ICCVW)},
	year     = {2023}
}

"""

__author__ = "Patrick Müller (2023)"


# -*- coding: utf-8 -*-
# based on:
#retrain_dnn.ipynb
# Retrain a DNN from pytorch
# based on: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os
import copy
from tqdm import tqdm
import sched, time
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from torchvision import models
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import utils # https://github.com/pytorch/vision/blob/1353df9e45aae6739c23db4a053c3d9a9fd17b3d/references/classification/

# each recipe contains of load_recipe function:
from recipes import efficientnet_b0, \
    efficientnet_b4,\
    mobilenet_v3_large,\
    resnet50,\
    resnet101,\
    resnext50_32x4d,\
    densenet161,\
    alexnet,\
    vgg16,\
    regnet_y_8gf,\
    squeezenet1_0



__recipes__ = [efficientnet_b0,efficientnet_b4,mobilenet_v3_large,resnet50,\
    resnet101,resnext50_32x4d,densenet161,alexnet,vgg16,regnet_y_8gf,squeezenet1_0]

__recipes__ = {r.__name__.split("recipes.")[1]: r.load_recipe for r in __recipes__}


__effective_batch_size__ = 256  # 8 V100 GPUs, bs = 32 -> 32*8 =256


import recipes._preprocessing as _preprocessing
from recipes._augment import OpticsAugment


def _get_model(name,device='cpu'):
    model_dnn = models.__dict__[name](pretrained=False)
    model_dnn.__name__ = name  # for internal use, overwrite later 
    model_dnn.train()
    model_dnn = model_dnn.to(device)    
    return model_dnn


def _get_savepath(fdir,fullname,ext):
    """ multiple days computation -> do not overwrite existing temp files"""
    savepath = os.path.join(fdir,fullname + ext)
    if os.path.exists(savepath):
        i = 1
        exists = True
        while exists:
            if os.path.exists(savepath):
                savepath = os.path.join(fdir, fullname + f"{i}" + ext)
                i+=1
            else:
               exists=False
    return savepath


def _getdataloaders(setup,optics_augment=True,augmix=False):
    """
    preprocessing and dataloading 
    """	
    if optics_augment and not augmix:
        preprocessing_fcn = _preprocessing.opticsaugment
    else:
        preprocessing_fcn = _preprocessing.augmix if augmix else _preprocessing.basic
        
    print(preprocessing_fcn)
    image_datasets,dataloaders = preprocessing_fcn(setup)

    return image_datasets,dataloaders


def _get_checkpoint_load_path(savename,setup):
    # get the load path for last available checkpoint:
    # use last existing checkpoint (_temp<i>.pt): make visible to checkpoint loader
    # in train() --> <model_name>_temp<i+1>.pt	
    model_path = _get_savepath(setup["model_dir"],f"{os.path.splitext(savename)[0]}_temp",".pt")
    next = model_path.split("_temp")[1].split(".pt")[0] # >=1 by design if _temp.pt exists 
    if next:
        next = int(next)
        if next==1:
            model_path = model_path.split("_temp")[0] + "_temp.pt"
        else: 
            load = next-1
            model_path = model_path.split("_temp")[0] + f"_temp{load}.pt"
    return model_path

def _remove_module_from_state_dict(state_dict):
    """
    #   module.features. ... 
    """
    keys = list(state_dict.keys())
    for key in keys:
        if "module." in key:
            new_key = key.replace("module.","")
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def _update_model_name_and_paths(setup,model_name,augmix=False,optics_augment=True):
    """
    model_name: no longer pytorch equivalent but specified by recipe --> change model.__name__
    updates setup <dict> 
    """
    savename = model_name + "{}{}.pt"
    savename = savename.format('_augmix' if augmix else '','_opticsaugment' if optics_augment else '')    
            
    setup["model_name"] = os.path.splitext(savename)[0]
    setup["savepath"] = os.path.join(setup["model_dir"],savename)
    setup["model_path"] = _get_checkpoint_load_path(savename,setup)
    return setup


def _get_recipe(setup,model_dnn,**kwargs): 
    """ recipe names: __recipes__ -> .__name__ 
    load and apply recipes to configuration 
    """
    name = model_dnn.__name__
    if name not in __recipes__:
        raise ValueError(f"training for model {name} not implemented yet.")        
    return __recipes__[name](model_dnn,setup=setup,**kwargs)
		

def _get_train_config(setup,model_dnn,optics_augment=True,augmix=False,**kwargs):
    """
    load_recipe from /recipes and configure training setup 
    
    contains optimizer, lr_scheduler and other training setup <dict>     
    """		
    setup, model_dnn, model_name = _get_recipe(setup,model_dnn,**kwargs)        
    setup = _update_model_name_and_paths(setup,model_name,augmix=augmix,\
        optics_augment=optics_augment)

    model_dnn.__name__ = setup["model_name"]

    return setup, model_dnn


def train(model, dataloaders, image_datasets,device="cpu",savepath="",\
        num_epochs=25,start_at_epoch=0,scheduler=None,criterion=None, optimizer=None,\
        batch_size=None,optics_augment=None,**kwargs):
    """! 
    modified from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#
    
    train a model for one epoch 
    
    model: 
    dataloaders: 
    image_datasets: <dict> , (train,val)
    savepath:
    num_epochs
    ...
    batch_size: <int>, just for printing
    optics_augment: BlurAugment object to apply the augmentation 
    
    """
    if optics_augment is None and kwargs.get("blur_augment",None):
        optics_augment = kwargs["blur_augment"]  # version compatibility 
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    since = time.time()
    fdir,name = os.path.split(savepath)
    name,ext = os.path.splitext(name)
    savepath_epoch = _get_savepath(fdir,name + "_temp", ext)
    savepath_epoch_curves = _get_savepath(fdir,name + "curves_temp",ext)
    savepath_curves = os.path.join(fdir,f"curves_{name}{ext}")
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = nn.DataParallel(model).cuda() # Distribute model across gpus 
    cudnn.benchmark = True

    loss_train_curve = [None]*num_epochs
    acc_train_curve = [None]*num_epochs
    loss_val_curve = [None]*num_epochs
    acc_val_curve = [None]*num_epochs

    assert start_at_epoch < num_epochs, \
        f"Neural Network already trained for all {num_epochs} epochs, stop."

    for epoch in range(start_at_epoch,num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, targets in dataloaders[phase]:
                if phase == "train" and kwargs.get("pipelined",None):
                    inputs,probabilities = inputs
                    probabilities = torch.Tensor(probabilities)
                else:
                    probabilities = None
                    
                inputs = inputs.to(device)
                targets = targets.to(device)

                if phase == "train" and optics_augment is not None:
                    inputs = optics_augment._aug_batch(inputs,device=device,weights=probabilities)
                    
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train': # this is the actual training --
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == targets.data)

            if scheduler is not None and phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and (epoch_acc > best_acc or epoch == 0):
                best_acc = epoch_acc
                best_epoch = epoch
                best_epoch_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                loss_train_curve[epoch] = epoch_loss
                acc_train_curve[epoch] = epoch_acc
            if phase == 'val':
                loss_val_curve[epoch] = epoch_loss
                acc_val_curve[epoch] = epoch_acc

        # deep copy the model
        torch.save([loss_train_curve,loss_val_curve,acc_train_curve,acc_val_curve],\
            savepath_epoch_curves)

        time_elapsed = time.time() - since
        print(f'epoch completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        print(f"write temporary model at {epoch} to {savepath_epoch}.")
        torch.save({'epoch': epoch,
                    'stats':{'batch_size':batch_size,'time_per_epoch':time_elapsed},
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_epoch_loss,
                    'acc': best_acc
                    }, savepath_epoch)


    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights and save:
    model.load_state_dict(best_model_wts)
    torch.save({
                'epoch': num_epochs,
                'best_epoch':best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_epoch_loss,
                'acc':best_acc
                }, savepath)
    # save final learning curve:
    torch.save([loss_train_curve,loss_val_curve,acc_train_curve,acc_val_curve],savepath_curves)


    curves = [loss_train_curve,loss_val_curve,acc_train_curve,acc_val_curve]
    return model,best_epoch_loss,best_epoch, curves


def main(name="efficientnet_b0",clean=True,augmix=False,optics_augment=True,**kwargs):
    """!
    name: model name, see list below or at recipes 
    clean: train on clean data or on imagenet-100-optics (clean + param 4, sev 1-5)
    
    tbd: extend by transformer networks, and ConvNeXt    
    """
    default_dir = r"S:\Framework_Users\NeuralNetsProjects"
    print(f"\n\nNow training {name}\n\n")

    if kwargs.get('root_dir',None) is None:
        kwargs.pop('root_dir')
    if kwargs.get('model_dir',None) is None:
        kwargs.pop('model_dir')
    
    setup = {}
    #setup["ann_path"] = os.path.join(setup["root_dir"],'annotations_val_2012_classes.json')
    if clean:
        setup["root_dir"]  = kwargs.get('root_dir',os.path.join(default_dir,\
            r"imagenet-100_optics_new\imagenet100_clean"))
        setup["model_dir"] = kwargs.get('model_dir',os.path.join(default_dir,\
			"models",os.path.split(setup['root_dir'])[1]))
    else:
        setup["root_dir"]  = kwargs.get('root_dir',os.path.join(default_dir,\
            r"ImageNet-100_optics_dataset/ImageNet100-optics"))
        setup["model_dir"] = kwargs.get('model_dir',os.path.join(default_dir,\
			"models",os.path.split(setup['root_dir'])[1]))
    
    print(f"Now training on {os.path.split(setup['root_dir'])}\nmodel_dir: {setup['model_dir']}")
    
    if not os.path.exists(setup["model_dir"]):
        os.makedirs(setup["model_dir"])
    
    setup["train_dir"] = os.path.join(setup["root_dir"],"train")
    setup["val_dir"]   = os.path.join(setup["root_dir"],"val")
    isgpu = torch.cuda.is_available()
    
    # https://stackoverflow.com/questions/62907815/pytorch-what-is-the-difference-between-tensor-cuda-and-tensor-totorch-device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if "mobilenet_large_v3" in name:
        kwargs["take_params"] = "kuan_wang_warm_restarts"

    model_dnn = _get_model(name,device=device)
    
    print(f"\nModel: {name}")
    
    setup, model_dnn = _get_train_config(setup,model_dnn,optics_augment=optics_augment,\
	    augmix=augmix,**kwargs)


    setup["training"]["pipelined"] = True if (optics_augment and augmix) else False
    setup["training"]["optics_augment"] = OpticsAugment() if optics_augment else None

    image_datasets,dataloaders = _getdataloaders(setup,optics_augment=optics_augment,\
        augmix=augmix)


    if setup["model_path"] is not None and os.path.exists(setup["model_path"]):
        checkpoint = torch.load(setup["model_path"])
        epoch = checkpoint['epoch']
        setup["training"]["start_at_epoch"] = epoch
        optimizer = setup["training"]["optimizer"]        
        model_dnn.load_state_dict(_remove_module_from_state_dict(checkpoint['model_state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model checkpoint at epoch #{epoch} from {setup['model_path']}")

    if not os.path.exists(setup["model_dir"]):
        os.makedirs(setup["model_dir"])
        print("Created new {setup['model_dir']}")

    # now adjust learning rate to effective batch_size: (June 16,2023)
    effective_batch_size = kwargs['num_gpus'] * setup['training']['batch_size']

    for i,__ in enumerate(setup['training']['optimizer'].param_groups):
        learning_rate = setup['training']['optimizer'].param_groups[i]['lr']
        learning_rate *= (effective_batch_size / __effective_batch_size__)
        setup['training']['optimizer'].param_groups[i]['lr'] = learning_rate
        print(f"changed learning rate {i} to: {learning_rate} (effective batch_size {effective_batch_size}"+\
            ", default: {__effective_batch_size__})")

    print(f"creating: {setup['savepath']}")
    print(f'\ntraining setup:\n{setup["training"]}\n')

    train(model_dnn,dataloaders, image_datasets,device=device,\
            savepath=setup['savepath'],**setup['training'])


if __name__ == "__main__":
	# store_true: default=False , store_false: default=True
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",default="efficientnet_b0",\
        help="type the name of the neural network to train")
    parser.add_argument("-imagefolderpath","--root_dir",default=None,type=str,\
        help="set the path of the imagenet dataset containing two folders: train, val")
    parser.add_argument("-modelpath","--model_dir",default=None,type=str,\
        help="set the savepath for the model checkpoints")
    
    parser.add_argument("--num_workers",default=8,type=int,\
        help="set the number of workers num_worker of the dataloader <= num_cpus (cores)")
    parser.add_argument("--batch_size",default=32,type=int,\
        help="set the batch size of the dataloader")
    parser.add_argument("--num_gpus",default=1,type=int,\
        help="set the number of gpus")
    parser.add_argument("--num_epochs",default=90,type=int,\
        help="set the number of epochs for training")

    parser.add_argument("--augmix",action="store_true",default=False)
    parser.add_argument("-nobluraugment","--bluraugment_off",action="store_true")
    args = parser.parse_args().__dict__

    print(f"{''.join(['#']*10)} This is a single GPU script {''.join(['#']*10)}")


    main(clean=True,optics_augment=not args['bluraugment_off'],**args)

# EOF
