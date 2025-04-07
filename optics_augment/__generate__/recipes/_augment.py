# 13.02.2023  --> augment training data from imagenet (ilsvrc2012)
import os 
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import sys
#import pdb # python debugger
import random
import time
import argparse
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms,datasets
from torch.utils.data import DataLoader,Dataset
import PIL 
import matplotlib.pyplot as plt


from ._preprocessing import imagenet_normalization


class ImageNetDataset(Dataset):
	""" 
	ImageNetDataset as a base class 
	"""
	def __init__(self,dataset,preprocess):
		self.dataset = dataset 
		self.classes = dataset.classes
		self.preprocess = preprocess
		self.index = 0
		
	def __getitem__(self,index):
		img,target = self.dataset[index]
		self.index = index
		return self.preprocess(img),target
	
	def __next__(self):
		self.index += 1
		return self.__getitem__(self.index)
	
	def __len__(self):
		return len(self.dataset)
from PIL import Image
import matplotlib.pyplot as plt

from torch.nn.parallel import parallel_apply



class OpticsAugment():
	"""
	OpticsAugment using Zernike Polynomials, ... phase: content preparation (confidential), Patrick M. 2023
	"""
	__parameters__ = [3,4,5,6,7,8,9,10]
	__severities__ = [1,2,3,4,5]
	
	def __init__(self,path_to_psf_stack=None,severity=3,alpha=1.0,normalize=True):
		"""!
		normalize: 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        (applied after augmentation step)
		"""
		self.psf_stack = self._load_blur_kernel_stack(path_to_psf_stack)
		
		assert self.psf_stack.shape[0] == len(self.__parameters__), ""
		
		self.alpha = alpha
		self.severity = severity 
		self.nparams = len(self.__parameters__)
		self.nsevs = len(self.__severities__)
		self.nchroma = self.psf_stack.shape[2] # 1, 5 or 10 
		self.normalize = imagenet_normalization() if normalize is not None else lambda x: x


	@staticmethod
	def _load_blur_kernel_stack(psfstack_path=None):
		"""!
		# now:  (<param>,<severity>,<chroma>,<color>,<psfy>,<psfx>)
				if ..ndim==5: 
					kernels.unsqueeze(2).shape: ([8, 5, 1, 3, 25, 25])
				else: shape: [8,5,nchroma,3,25,25]
		#.permute(0,1,3,4,2) #,dtype=torch.float32) -- no longer necessary, has the standard shape
		#  kernels.sum(axis=(-1,-2)) not working, so normalize to l1 this way:
		#p,s,c,__,__ = kernels.shape
		"""  
		if psfstack_path is None and not __name__ == "__main__":
			psfstack_path = os.path.abspath(os.path.join(os.getcwd(),"..","kernel_stack_iccvw.pt"))
		kernels = torch.load(psfstack_path)
		kernels = kernels.cuda() if torch.cuda.is_available() else kernels
		is_gpu = "gpu" if torch.cuda.is_available() else "cpu"
		if kernels.ndim == 5:
			kernels = kernels.unsqueeze(2)  # downward compatibility
		print(f"loaded kernel stack of shape: {kernels.shape} ({is_gpu}),\n{psfstack_path}")
		return kernels
		

	def _aug_batch_one_kernel(self,images,chroma=True):
		"""!
		use a single kernel per batch, single weight
		"""
		_batchsize = images.shape[0]
		_param = int(random.random()*self.nparams)
		_chroma = int(random.random()*self.nchroma)
		weights = torch.Tensor([np.float32(np.random.beta(self.alpha,self.alpha)) \
			for _ in range(_batchsize)])				
		self.param = self.__parameters__[_param] # for debugging the param
		kernel = self.psf_stack[_param, self.severity-1, _chroma,:].unsqueeze(0)
		return (1 - weights.view(-1,1,1,1)) * images + weights.view(-1,1,1,1) * self._apply(images,kernel)


	def _aug_batch(self,images,chroma=True,device=None,weights=None):
		"""!
		applies the OpticsAugment to an image batch... 
		#need to be normalized afterwards? Yes.
		"""
		_batchsize = images.shape[0]
		_chroma = int(random.random()*self.nchroma)
		_params = [random.choice(self.__parameters__)-3 for n in range(_batchsize)]
		self.param = _params
		if weights is None:
			weights = torch.Tensor([np.float32(np.random.beta(self.alpha,self.alpha)) \
				for _ in range(_batchsize)])

		weights = weights.to(device)
		kernels = self.psf_stack[:, self.severity-1, _chroma,:]		
		blurred_ims = torch.stack(\
			parallel_apply([self._apply]*len(images),\
			[(im,kernels[_param,:].unsqueeze(0)) for im,_param in zip(images,_params)]),\
			dim=0).squeeze()

		return self.normalize((1 - weights.view(-1,1,1,1)) * images + weights.view(-1,1,1,1) * blurred_ims)


	@staticmethod
	def _aug_batch_deterministic(images,psf_stack,params=None,severity=3):
		"""!"""
		raise NotImplementedError("removed")


	@staticmethod
	def _apply(img,kernel,devices=None): # should work for img_batch also 
		#### this should be executed on batches at once, if possible:
		"""
		Args:
			batch <torch.tensor>: (minibatch,in_channels,iH,iW)
			--> (batchsize,rgb,iH,iW): (100,3,224,224)
			kernel <torch.tensor>: (out_channels, in_channels / group ,kH,kW)
			--> (rgb,rgb,kH,kW): (3,3,25,25)	
			padding='same', dilation = 1 only for uneven sized kernels
		"""		
		if img.shape[0] in [1,3]:
			img = img.unsqueeze(0) # color image? 
		if kernel.ndim == 3:
			kernel = kernel.unsqueeze(0)

		#try:
		ncol = kernel.shape[-3]
		return torch.stack(parallel_apply([OpticsAugment._conv2d]*ncol,\
			[(img[:,ch,:,:].unsqueeze(1),kernel[:,ch,:,:].unsqueeze(1)) for ch in range(ncol)])\
			,dim=0)
			

	@staticmethod
	def _get_param_and_severity_label(param,severity):
		return param+3, severity+1


	@staticmethod
	def _get_param_and_severity_idx(param,severity):
		return param-3, severity-1



	@staticmethod
	def _conv2d(batch,kernel,padding_mode="constant"):
		"""!
		Args:
			batch <torch.tensor>: (minibatch,in_channels,iH,iW)
			--> (batchsize,1,iH,iW): (100,1,224,224)
			kernel <torch.tensor>: (out_channels, in_channels / group ,kH,kW)
			--> (1,1,kH,kW): (1,1,25,25)		
		"""
		p = kernel.shape[-1]//2
		return F.conv2d(F.pad(batch,(p,p,p,p),mode=padding_mode),
				  kernel,bias=None,stride=1,padding='valid',dilation=1,groups=1).squeeze()


BlurAugment= OpticsAugment  #alias 


def test_imagenet_dataset():
	path_to_images = r'/blur/ilsvrc2012_gt'
	preprocess = transforms.Compose([transforms.PILToTensor()])
	imagenet = ImageNetDataset(datasets.ImageFolder(path_to_images),preprocess)
	print(f"length: {imagenet.__len__()}")

	i = 0
	__next__ = True
	while __next__:
		if i == 0:
			item = imagenet.__getitem__(\
				int(imagenet.__len__()*random.random()))
		else:
			item = imagenet.__next__()
			
		plt.imshow(item['image'])
		plt.title(item['ann'])
		plt.show(block=False)
		if input("If no input... continue: "):
			__next__ = False
		plt.close()


if __name__ == "__main__":
	pass
	#test_imagenet_dataset()
	#test_random_aug()
	#test_speed_augmentation(path_to_images=impath)