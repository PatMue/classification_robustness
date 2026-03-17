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

import os 
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.parallel import parallel_apply


class OpticsAugment(torch.nn.Module):
	"""
	OpticsAugment using Wave Optics PSFs.
	"""
	__parameters__ = [3,4,5,6,7,8,9,10]
	__severities__ = [1,2,3,4,5]

	max_size = 37  #  25... -> 37
	

	def __init__(self,path_to_psf_stack=None,severity=3,alpha=1.0,normalize=None,
			  device='cpu',padding_mode='reflect',oa_loss=False):
		"""
		normalize: 
			a torch transform directly input for a specific model.  
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
		(applied after augmentation step)

		alpha: from symmetric beta distribution, default: 1.0 to have uniform distribution
		"""
		super().__init__()
		self.device = device 
		self.psf_stack = self._load_blur_kernel_stack(path_to_psf_stack,device=self.device)		
		assert self.psf_stack.shape[0] == len(self.__parameters__), ""

		self.alpha = alpha
		self.padding_mode = padding_mode	
		self.severity = severity 
		self.nparams = len(self.__parameters__)
		self.nsevs = len(self.__severities__)
		self.nchroma = self.psf_stack.shape[2] # 1, 5 or 10 
		self.normalize = normalize if normalize is not None else lambda x: x
		self.oa_loss = oa_loss
		self.max_size = self.psf_stack.shape[-1]


	@staticmethod
	def _load_blur_kernel_stack(psfstack_path=None,device='cpu'):
		"""!
		# (<param>,<severity>,<chroma>,<color>,<psfy>,<psfx>)
		"""  
		if psfstack_path is None and not __name__ == "__main__":
			psfstack_path = os.path.abspath(os.path.join(os.getcwd(),"..","kernel_stack_iccvw.pt"))
		kernels = torch.load(psfstack_path)
		kernels = kernels.to(device) if torch.cuda.is_available() else kernels
		is_gpu = "gpu" if torch.cuda.is_available() else "cpu"
		if kernels.ndim == 5:
			kernels = kernels.unsqueeze(2)  # downward compatibility
		print(f"loaded kernel stack of shape: {kernels.shape} ({is_gpu}),\n{psfstack_path}")
		return kernels


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
		if torch.cuda.is_available():
			blurred_ims = torch.stack(\
				parallel_apply([self._apply]*len(images),\
				[(im,kernels[_param,:].unsqueeze(0),self.padding_mode) for im,_param in zip(images,_params)]),\
				dim=0).squeeze()
		else:
			blurred_ims = torch.stack(\
				[self._cpu_apply(im,kernels[_param,:].unsqueeze(0),self.padding_mode) for im,_param in zip(images,_params)],\
				dim=0).squeeze()

		return self.normalize((1 - weights.view(-1,1,1,1)) * images + weights.view(-1,1,1,1) * blurred_ims)


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
		if torch.cuda.is_available():
			blurred_ims = self._apply(images,kernel,self.padding_mode)
		else:
			blurred_ims = self._cpu_apply(images,kernel,self.padding_mode)
		return self.normalize((1 - weights.view(-1,1,1,1)) * images + weights.view(-1,1,1,1) * blurred_ims)


	def forward(self,batch:torch.Tensor,targets:torch.Tensor,chroma=True,weights=None):

		if batch.ndim != 4:
			raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
		if not batch.is_floating_point():
			raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")

		if not batch.get_device()==self.device:
			batch = batch.to(self.device)
			targets = targets.to(self.device)

		_batchsize = batch.shape[0]
		_chroma = int(random.random()*self.nchroma) # deprecated... no use - replaced with KernelAugment
		_params = [random.choice(self.__parameters__)-3 for n in range(_batchsize)]
		self.param = _params
	
		if weights is None:
			weights = torch.Tensor([np.float32(np.random.beta(self.alpha,self.alpha)) \
				for _ in range(_batchsize)])

		weights = weights.to(self.device)
		kernels = self.psf_stack[:, self.severity-1, _chroma,:]		# [params,sev,chroma,3,25,25] -> [params,3,25,25] -> (8,3,25,25)
		kernels = torch.reshape(kernels,(-1,3,self.max_size,self.max_size))

		if torch.cuda.is_available():
			blurred_ims = torch.stack(\
				parallel_apply([self._apply]*len(batch),\
				[(im,kernels[_param,:].unsqueeze(0),self.padding_mode) for im,_param in zip(batch,_params)]),\
				dim=0).squeeze()
		else:
			blurred_ims = torch.stack(\
				[self._cpu_apply(im,kernels[_param,:].unsqueeze(0),self.padding_mode) for im,_param in zip(batch,_params)],\
				dim=0).squeeze()

		
		if self.oa_loss:
			return (self.normalize(batch), 
		   		(1 - weights.view(-1,1,1,1)) * self.normalize(batch) + weights.view(-1,1,1,1) * self.normalize(blurred_ims)), targets

		return (1 - weights.view(-1,1,1,1)) * self.normalize(batch) + weights.view(-1,1,1,1) * self.normalize(blurred_ims), targets


	@staticmethod
	def _apply(img,kernel,padding_mode): 
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

		ncol = kernel.shape[-3]
		return torch.stack(parallel_apply([OpticsAugment._conv2d]*ncol,\
			[(img[:,ch,:,:].unsqueeze(1),kernel[:,ch,:,:].unsqueeze(1),padding_mode) for ch in range(ncol)])\
			,dim=0)


	@staticmethod
	def _cpu_apply(img,kernel,padding_mode): # should work for img_batch also 
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

		ncol = kernel.shape[-3]
		return torch.stack([OpticsAugment._conv2d(img[:,ch,:,:].unsqueeze(1),
											kernel[:,ch,:,:].unsqueeze(1),padding_mode) \
											for ch in range(ncol)],dim=0)



	@staticmethod
	def _get_param_and_severity_label(param,severity):
		return param+3, severity+1


	@staticmethod
	def _get_param_and_severity_idx(param,severity):
		return param-3, severity-1


	@staticmethod
	def _conv2d(batch,kernel,padding_mode='reflect'):
		"""!
		#requires torch>=1.9.0
		# torch.backends.cudnn.deterministic = True (consider)	
		Args:
			batch <torch.tensor>: (minibatch,in_channels,iH,iW)
			--> (batchsize,1,iH,iW): (100,1,224,224)
			kernel <torch.tensor>: (out_channels, in_channels / group ,kH,kW)
			--> (1,1,kH,kW): (1,1,25,25)		
		"""
		if padding_mode == 'reflect':
			p = kernel.shape[-1]//2
			return F.conv2d(F.pad(batch,(p,p,p,p),mode=padding_mode),
					kernel,bias=None,stride=1,padding='valid',dilation=1,groups=1).squeeze()

		return F.conv2d(batch,kernel,bias=None,stride=1,padding='same',dilation=1,groups=1).squeeze()


if __name__ == "__main__":
	pass
	#test_imagenet_dataset()
	#test_random_aug()
	#test_speed_augmentation(path_to_images=impath)