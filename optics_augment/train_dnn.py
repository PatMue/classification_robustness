"""
If you find this useful in your research, please cite:

@article{mueller2023_opticsbench,
	author   = {Patrick Müller, Alexander Braun and Margret Keuper},
	title    = {Classification robustness to common optical aberrations},
	journal  = {Proceedings of the International Conference on Computer Vision Workshops (ICCVW)},
	year     = {2023}
}

"""

__author__ = "Patrick Müller"


# -*- coding: utf-8 -*-
# based on:
#retrain_dnn.ipynb
# Retrain a DNN from pytorch
# based on: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from glob import glob
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
from torchvision import models
import PIL
from PIL import Image

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

import preprocessing as preprocessing
from preprocessing import DefaultNormalize
from augment import OpticsAugment


__recipes__ = [efficientnet_b0,efficientnet_b4,mobilenet_v3_large,resnet50,\
    resnet101,resnext50_32x4d,densenet161,alexnet,vgg16,regnet_y_8gf,squeezenet1_0]

__recipes__ = {r.__name__.split("recipes.")[1]: r.load_recipe for r in __recipes__}
__effective_batch_size__ = 256  # 8 V100 GPUs, bs = 32 -> 32*8 =256


def _get_model(name,device='cpu'):
    model_dnn = models.__dict__[name](pretrained=False)
    model_dnn.__name__ = name  # for internal use, overwrite later 
    model_dnn.train()
    model_dnn = model_dnn.to(device)    
    return model_dnn


def _get_savepath(fdir, fullname, ext):
    """ multiple days computation -> do not overwrite existing temp files"""
    savepath = os.path.join(fdir, fullname + ext)
    i = 1
    while os.path.exists(savepath):
        savepath = os.path.join(fdir, f"{fullname}{i}{ext}")
        i += 1
    return savepath


def _getdataloaders(setup,optics_augment=True,augmix=False):
    """
    preprocessing and dataloading 
    """	
    if optics_augment and not augmix:
        preprocessing_fcn = preprocessing.opticsaugment
    else:
        preprocessing_fcn = preprocessing.augmix if augmix else preprocessing.basic
        
    print(preprocessing_fcn)
    image_datasets,dataloaders = preprocessing_fcn(setup)

    return image_datasets,dataloaders


def _get_checkpoint_load_path(savename, setup):
    base_name = os.path.splitext(savename)[0]
    search_pattern = os.path.join(setup["model_dir"], f"{base_name}_temp*.pt")
    
    existing_ckpts = glob.glob(search_pattern)
    
    if not existing_ckpts:
        return os.path.join(setup["model_dir"], f"{base_name}_temp.pt")
        
    latest_ckpt = max(existing_ckpts, key=os.path.getmtime)
    return latest_ckpt


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
    optics_augment: OpticsAugment object to apply the augmentation 
    """    

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

    model = model.to(device) 
    cudnn.benchmark = True

    loss_train_curve = [None]*num_epochs
    acc_train_curve = [None]*num_epochs
    loss_val_curve = [None]*num_epochs
    acc_val_curve = [None]*num_epochs

    assert start_at_epoch < num_epochs, \
        f"Neural Network already trained for all {num_epochs} epochs, stop."
    
    best_epoch = start_at_epoch
    best_epoch_loss = torch.inf

    for epoch in range(start_at_epoch,num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

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

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

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
                    }, 
                    savepath_epoch
                )


    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    torch.save({
                'epoch': num_epochs,
                'best_epoch':best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_epoch_loss,
                'acc':best_acc
                }, savepath)
    torch.save([loss_train_curve,loss_val_curve,acc_train_curve,acc_val_curve],savepath_curves)

    curves = [loss_train_curve,loss_val_curve,acc_train_curve,acc_val_curve]
    return model,best_epoch_loss,best_epoch, curves


def main(args):
    """!
    name: model name, see list below or at recipes 
    clean: train on clean data or on imagenet-100-optics (clean + param 4, sev 1-5)    
    TODO: extend by transformer networks, and ConvNeXt    
    """
    args.optics_augment = not args.opticsaugment_off

    setup = {}
    setup["root_dir"]  = args.root_dir
    setup["model_dir"] = args.model_dir 
    kwargs = args.__dict__

    print(f"Now training {args.model} on {setup['root_dir']}\nmodel_dir: {setup['model_dir']}")
    
    if not os.path.exists(setup["model_dir"]):
        os.makedirs(setup["model_dir"])
    
    setup["train_dir"] = os.path.join(setup["root_dir"],"train")
    setup["val_dir"]   = os.path.join(setup["root_dir"],"val")
    
    device = torch.device(args.device) if torch.cuda.is_available() else "cpu"
        
    if "mobilenet_large_v3" in args.model:
        kwargs["take_params"] = "kuan_wang_warm_restarts"

    model_dnn = _get_model(args.model,device=device)
    
    print(f"\nModel: {args.model}, optics_augment: {args.optics_augment}, augmix: {args.augmix}\n")
    
    setup, model_dnn = _get_train_config(setup,model_dnn,**args.__dict__)

    setup["training"]["pipelined"] = True if (args.optics_augment and args.augmix) else False

    if args.optics_augment:
        setup["training"]["optics_augment"] = OpticsAugment(
                                        severity=args.opticsaugment_severity,
                                        alpha=args.opticsaugment_alpha,
                                        normalize=DefaultNormalize.get(),
            		                	device=args.device,
                                        ) 
    else:
        setup["training"]["optics_augment"] = None

    image_datasets,dataloaders = _getdataloaders(setup,optics_augment=args.optics_augment,augmix=args.augmix)

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
        print(f"Created new {setup['model_dir']}")

    effective_batch_size = args.num_gpus * setup['training']['batch_size']

    for i,__ in enumerate(setup['training']['optimizer'].param_groups):
        learning_rate = setup['training']['optimizer'].param_groups[i]['lr']
        learning_rate *= (effective_batch_size / __effective_batch_size__)
        setup['training']['optimizer'].param_groups[i]['lr'] = learning_rate
        print(f"changed learning rate {i} to: {learning_rate} (effective batch_size {effective_batch_size}"+\
            f", default: {__effective_batch_size__})")

    print(f"creating: {setup['savepath']}")
    print(f'\ntraining setup:\n{setup["training"]}\n')


    train(model_dnn,dataloaders, image_datasets,device=device,\
            savepath=setup['savepath'],**setup['training'])


if __name__ == "__main__":
    # python train_dnn.py --model squeezenet1_0 --root_dir /path/to/imagenet/ --model_dir /path/to/save/model/
    parser = argparse.ArgumentParser()

    parser.add_argument("-m","--model",default="efficientnet_b0",help="model name")
    parser.add_argument("-imagefolderpath","--root_dir",default=None,type=str,\
                    help="imagenet dataset path containing two folders: train, val")
    parser.add_argument("--model_dir",default=None,type=str,help="savepath for model checkpoints")
    
    parser.add_argument("--num_workers",default=8,type=int,\
        help="set the number of workers num_worker of the dataloader <= num_cpus (cores)")
    parser.add_argument("--batch_size",default=32,type=int,\
        help="set the batch size of the dataloader")
    parser.add_argument("--num_gpus",default=1,type=int,help="set the number of gpus") # FIXME: use DistributedDataParallel
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str, help="device to train on")
    parser.add_argument("--num_epochs",default=90,type=int,\
        help="set the number of epochs for training")
    
    parser.add_argument("--opticsaugment_off",action="store_true",help='turn off opticsaugment for training')
    parser.add_argument("--opticsaugment_severity",default=3,type=int,help="severity of optics augment, 1-5")
    parser.add_argument("--opticsaugment_alpha",default=1.0,type=float,help="alpha of optics augment, from symmetric beta distribution, default: 1.0 to have uniform distribution")

    parser.add_argument("--augmix",action="store_true",default=False)
    args = parser.parse_args()

    print(f"{''.join(['#']*10)} This is a single GPU script {''.join(['#']*10)}")

    main(args)

# EOF
