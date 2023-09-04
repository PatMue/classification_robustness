# update 20.06.2023, Patrick Müller 2023
import os
import argparse 
import platform 

import numpy as np
import torch
import matplotlib.pyplot as plt

"""
example usage: 

python3 vis_learning.py -n resnet50_sgd -d imagenet-1k-split --root_dir ~/Documents/omni/download/models/

"""
#__ROOT_DIR__ = r"S:\Framework_Users\NeuralNetsProjects\models"

def load_curves(dataset="",curves_name="curves",root_dir=None,**kwargs):
    """!"""

    if kwargs.get('merge',None) is not None:
        curves = merge_curves(dataset=dataset,curves_name=curves_name,
                              model_name=kwargs.get('model_name',None),
                              root_dir=root_dir)
    else:
        curves = torch.load(os.path.join(root_dir,dataset,f"{curves_name}.pt"),map_location=\
            torch.device('cpu'))    
        print(curves)

    curves = [[cc.cpu() if isinstance(cc,torch.Tensor) else cc for cc in c]for c in curves]
    return curves


def merge_curves(dataset="",curves_name="curves",model_name="",root_dir=None,**kwargs):
    """
    # --root_dir ~/Documents/omni/download/models --dataset imagenet-1k_new --model_name resnet50_sgd --merge
    merge _temp files to display whole learning curve for training 
    """

    def _get_valid_indices(list1):
        return [i for i,c in enumerate(list1) if c is not None]

    if not model_name is None:
        curves_names = [f for f in os.listdir(os.path.join(root_dir,dataset)) \
                        if f.__contains__(model_name + 'curves')] # directly append 'curves' as identifier

        print(f"\nmerged files: {curves_names}\n\n") #--> now okay 

    curves_list = [] 
    for i,curves_name in enumerate(curves_names):
        curves_temp = torch.load(os.path.join(root_dir,dataset,f"{curves_name}"),map_location=\
            torch.device('cpu'))
        curves_list.append(curves_temp)

    maxid = [len(c[0]) for c in curves_list]
    maxid = [i for i,c in enumerate(maxid) if c == max(maxid)][0] 
    maxlen = len(curves_list[maxid][0]) # for multiple entries use the first max

    curves = np.array(curves_list[maxid])

    for i,curves_temp in enumerate(curves_list):
        if i != maxid:
            for c,curve_temp in enumerate(curves_temp):
                idx = _get_valid_indices(curve_temp)
                curves[c,idx] = [curve_temp[id] for id in idx]

    return curves 


    
def plot_curves(curves,model_name="<model>",dataset="<dataset>",save=False,**kwargs):
    """!"""
    ttl = "Loss and Accuracy for Training and Validation"
    ttl += f"\n{model_name} on {os.path.split(dataset)[1]}"

    fig,ax = plt.subplots(1,2)

    ax[0].semilogy(curves[0],label="loss - train")
    ax[0].semilogy(curves[1],label="loss - val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    
    ax[1].plot([100*c for c in curves[2] if c is not None],label="acc - train")
    ax[1].plot([100*c for c in curves[3] if c is not None],label="acc - val")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy [%]")
    ax[1].legend()
    plt.grid()
    plt.suptitle(ttl)
    
    if platform.system() == 'Linux':
        plt.show()
    else:
        plt.show(block=False)
        input()
        plt.close()
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model name for vis of learning rate")
    parser.add_argument('--root_dir',default=None,help='set rootdir to models')
    parser.add_argument('-n','--model_name',default=None,help='set model name')
    parser.add_argument('--merge',action='store_true',help='combine temp results to one learning curve')
    parser.add_argument('-d','--dataset',default=None,help='set folder name')
    kwargs = parser.parse_args().__dict__
    plot_curves(load_curves(**kwargs),**kwargs)
    
    # python3 vis_learning.py --root_dir /models --dataset imagenet-1k_new --model_name resnet50_sgd_opticsaugment --merge
