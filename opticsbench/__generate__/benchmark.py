#################################
# (c) Patrick Müller 2023 
#################################


"""
If you find this useful in your research, please cite:

@article{mueller2023_opticsbench,
	author   = {Patrick Müller, Alexander Braun and Margret Keuper},
	title    = {Classification robustness to common optical aberrations},
	journal  = {Proceedings of the International Conference on Computer Vision Workshops (ICCVW)},
	year     = {2023}
}

"""
__license__ = "MIT-license"
__author__ = "Patrick Müller"

import os
import shutil
import sys
import random
import argparse
import time
import copy
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import json
import datetime
import logging
import gc

import torch
import torchvision
from torch.nn.parallel import parallel_apply
from torchvision import datasets,transforms,models
from PIL import Image


import matplotlib.pyplot as plt 

#from robustbench.model_zoo.imagenet import imagenet_models  # switched to manual download since there are bugs in current version of robustbench repo 


import utils

sys.path.append(os.path.abspath(os.path.join("..","..","optics_augment","__generate__")))

from recipes._augment import BlurAugment, ImageNetDataset
import __registered_model_lists__

logger = logging.getLogger(__name__)


if torchvision.__version__ >= "0.11.0":
	__models__ = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'squeezenet1_0', 'squeezenet1_1', 'inception_v3', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',  'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf','regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf']
		

if torchvision.__version__ >= '0.12.0':
	# https://pytorch.org/vision/0.12/models.html , https://pytorch.org/vision/0.11/models.html
	__models__.extend(['vit_b_16','vit_b_32','vit_l_16','vit_l_32',\
		'convnext_tiny','convnext_small','convnext_base','convnext_large'])
# vision transformers and convnext is available from torchvision 0.12 ... (requires python >= 3.7, so not availble for training)


class ImageNetDatasetConverted(torch.utils.data.Dataset):
	"""
	ImageNetDataset as a base class

	convert: <dict> containing input and output dataset names
	"""
	def __init__(self,dataset,preprocess,convert=None):
		self.dataset = dataset
		self.classes = dataset.classes
		self.idx_to_class = {v: k for k,v in self.dataset.class_to_idx.items()}
		self._to_idx = self._class_to_idx(convert) # new dict
		self.preprocess = preprocess

	def _class_to_idx(self,convert):
		return utils.convert_imagenet1k_targets_to_x(**convert)

	def _get_new_target(self,target):
		cls = self.idx_to_class[target]
		return self._to_idx[cls]


	def __getitem__(self,index):
		img,target = self.dataset[index]
		return self.preprocess(img),self._get_new_target(target)

	def __len__(self):
		return len(self.dataset)


class ConvertTarget():
	def __init__(self,database,class_to_idx=None,convert=None):
		self.name = database.lower()
		if convert:
			if self.name.__contains__("imagenet-100") or self.name.__contains__("imagenet100"):
				convert = {"input":"ImageNet-1k","output":"ImageNet-100"}
			if self.name.__contains__("imagenette"):
				convert = {"input":"ImageNet-1k","output":"ImageNette"}
		self.convert = convert
		self.class_to_idx = class_to_idx
		self.idx_to_class = {v: k for k,v in self.class_to_idx.items()}
		self._to_idx = self._class_to_idx(convert) # new dict
		self.__true__ = True if self.convert and self.class_to_idx else False

	def _class_to_idx(self,convert):
		return utils.convert_imagenet1k_targets_to_x(**convert)

	def _get_new_target(self,target:int):
		cls = self.idx_to_class[target]
		return self._to_idx[cls]

	def _get_new_target_batch(self,targets):
		for i,target in enumerate(targets):
			targets[i] = self._get_new_target(target.item())
		return targets


"""

1. Assign paths
2. Run inference
3. save results
4. report results

(re-use existing code if available)

* do inference on simrechner (also colab possible if required, but base on simrechner)


# ImageNette/val Benchmark
# ImageNet-100/val Benchmark
# ImageNet-1k/val Benchmark

# develop everything for ImageNette, then do the same on the other ones

# create dataloader for each folder:

data/
	images
		ImageNette
			/val
			/corruptions
				opticsblur
						astigmatism (param 4,5)
							1
							2
							3
							4
							5
						coma (param 6,7)
							1
							2
							...
						trefoil  (param 9,10)
							...
						defocus_spherical (param 3,8)
							...
				common2d
						defocus_blur
		ImageNet-100
			/val
			/corruptions
			...
		ImageNet-1k
			/val
			/corruptions
			...

	eval
		ImageNette
			/val
			/corruptions
				opticsblur
						astigmatism (param 4,5)
							resnet50_sev1
							efficientnet_b0_sev1
							convnext_xy_sev1
							...
							resnet50_sev2
							...
						coma (param 6,7)
							resnet50
							efficientnet_b0
							convnext_xy
							...
						trefoil  (param 9,10)
							resnet50
							efficientnet_b0
							convnext_xy
							...
						defocus_spherical (param 3,8)
							resnet50
							efficientnet_b0
							convnext_xy
							...
				common2d
						defocus_blur
							resnet50
							efficientnet_b0
							convnext_xy
							...
		ImageNet-100
			/val
			/corruptions
			...

		ImageNet-1k
			/val
			/corruptions
			...

	models (or any setup, if required)
		resnet50_augmix_blur_augment
		efficientnet_b0_augmix_blur_augment
		convnext_xy_augmix
		...
	...


# 	evaluate image folder in images/ for one epoch,
	save results to same hierarchy: eval/

#   assumes that images are already generated and above folders exist? No, can create here using torch conv2d, using blur_stack only

"""


# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
# decorator used to block function printing to the console
def block_printing(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

@block_printing
def get_model_pretrained(fcn):
	return fcn(pretrained=True)

class EvalOpticsBenchmark():

	def __init__(self,path_to_root,path_to_images_folder,path_to_user_models="",\
			batch_size=25,num_workers=4,quiet=False):
		"""

		take a neural network, inference on dataset, save results to mirrored folder


		example usage:

		evalBench = EvalOpticsBenchmark(path_to_eval_folder,path_to_images_folder)

		model = evalBench._load_model('resnet50')
		database = 'ImageNette'
		mode = 'coma'
		severity = '2'
		split = 'corruptions'


		example usage:
		***********************************************************************
		acc = evalBench._run_inference_on_folder(model,database,mode,severity,split=split,\
			device='cuda')

		evalBench._report(acc,model,database,mode,severity,split=split)
		*************************************************************************
		"""
		self.paths = {"eval":os.path.join(path_to_root,"eval"),"images":path_to_images_folder,\
			"models":path_to_user_models,"log":os.path.join(path_to_root,"eval","logfile.log")}
		self.isgpu = torch.cuda.is_available()
		self.batch_size = batch_size
		self.num_workers = num_workers

		if not os.path.exists(self.paths["eval"]):
			# get list of imagefolders to be evaluated...
			self._mirror_folders()


		if quiet:
			self._get_logfile()
		self.quiet = quiet

		self.databases 	= self._get_available_datasets()
		self.models 	= self._get_available_user_models()
		self.log(f"databases: {self.databases}")
		self.log(f"models: {self.models}")


	def _get_logfile(self):
		"""
		Run once per class initialization, saved to eval folder top level
		"""
		exists = True
		i = 0
		while exists:
			if os.path.exists(self.paths["log"]):
				self.paths["log"] = os.path.join(self.paths["eval"],f"logfile_{i}.log")
				i += 1
			else:
				exists = False
		with open(self.paths['log'],"w") as f:
			f.write(f"This is the log file for evaloptics on {str(datetime.datetime.now())}")


	def _mirror_folders(self):
		"""
		mirror the tree of 'images' (without copying files)
		"""
		def ig_f(dir, files):
			return [f for f in files if os.path.isfile(os.path.join(dir, f)) or f.startswith("n0") or f.startswith("n1")]
		try:
			shutil.copytree(self.paths["images"],self.paths["eval"],ignore=ig_f)

		except FileExistsError as err:
			print(err)
			print("now writing to ... ")
			self.paths["eval"] = self.paths["eval"] + "_1"
			shutil.copytree(self.paths["images"],self.paths["eval"],ignore=ig_f)

		print(f"DONE, copying tree to: {self.paths['eval']}")


	def _mirror_subfolder(self,subfolder):
		"""
		mirror the tree of 'images' (without copying files)
		"""
		def ig_f(dir, files):
			return [f for f in files if os.path.isfile(os.path.join(dir, f)) or f.startswith("n0") or f.startswith("n1")]
		try:
			src = os.path.join(self.paths["images"],subfolder)
			dest = os.path.join(self.paths["eval"],subfolder)
			shutil.copytree(src,dest,ignore=ig_f)
			print(f"DONE, copying tree to: {self.paths['eval']}")

		except FileExistsError as err:
			pass #print(err)



	def _get_available_datasets(self):
		"""
		returns a <dict> containing all available datasets
		"""
		databases = {d:{"path":os.path.join(self.paths["images"],d),"subsets":{}} \
			for d in os.listdir(self.paths["images"])}

		for d in databases:
			fpath = databases[d]["path"]
			splits = [m for m in os.listdir(fpath) if os.path.isdir(os.path.join(fpath,m))]
			for split in splits:
				if split == "corruptions":
					subsets = {mode:{sev:os.path.join(fpath,split,mode,sev) for sev in os.listdir(\
						os.path.join(fpath,split,mode)) \
						if os.path.isdir(os.path.join(fpath,split,mode))} \
						for mode in os.listdir(os.path.join(fpath,split)) \
							if os.path.isdir(os.path.join(fpath,split))}
					#subsets = {s:subsets[s] if len(subsets[s]) else None for s in subsets}
					databases[d]["subsets"][split] = subsets
				elif split == "corruptions_rg":
					subsets = {mode:{sev:os.path.join(fpath,split,mode,sev) for sev in os.listdir(\
						os.path.join(fpath,split,mode)) \
						if os.path.isdir(os.path.join(fpath,split,mode))} \
						for mode in os.listdir(os.path.join(fpath,split)) \
							if os.path.isdir(os.path.join(fpath,split))}
					#subsets = {s:subsets[s] if len(subsets[s]) else None for s in subsets}
					databases[d]["subsets"][split] = subsets
				elif split == "val":
					databases[d]["subsets"][split] = os.path.join(fpath,split)
		return databases


	def _get_path_from_database(self,database,split=None,mode=None,sev=None):
		"""!
		database: key from self.databases
		returns filedirectory to imagedataset
		kwargs:
			split,
			mode,
			sev
		"""
		# search in the databases <dict> for above keys and return the path:
		database = self.databases.get(database,None)
		if database is None:
			raise KeyError(f"specified database not found in available databases: {database}")


		if split == "val":
			return database["subsets"][split]
		else:
			self.log(f"split: {split}, {mode}, {sev}")
			assert mode is not None, f"requires mode - corruption, but is: {mode}"
			assert sev in [1,2,3,4,5], f"mode requires 'sev', but is: {sev},{type(sev)}"
			# now it is in the keys
			try:
				return database["subsets"][split][mode][str(sev)]
			except KeyError:
				return database["subsets"][split][mode][sev]


	def _get_available_user_models(self):
		"""
		get list of user trained models (requires torchvision eqivalent description)
		"""

		def get_name_from_savepath(p):
			# prerequisite: needs to be in torchvision.models listed
			# find name in __models__
			name =  [model for model in __models__ if model in p]
			if len(name) == 1:
				return name[0]
			raise ValueError(f"Model not found in __models__: {p}")


		if os.path.exists(self.paths['models']):
			return {os.path.splitext(p)[0]:{"path": \
				os.path.join(self.paths["models"],p), "name":get_name_from_savepath(p)} for p in \
				os.listdir(self.paths["models"]) if p.endswith('.pt')}
		return {}


	@block_printing
	def _load_model(self,model_name:str,weights=None):
		"""

		model_name:  <str>
			either from description (requires to contain a torchvision model equivalent)
			or: torchvision model name (as in __models__)

		if model_name in available models (models/):
			select and load it  -- e.g.  resnet50_augmix_blur_augment
		else:
			try to find it in torchvision models
		"""
		from collections import OrderedDict
		def rm_substr_from_state_dict(state_dict, substr):
			# adapted from  https://github.com/RobustBench/robustbench/blob/master/robustbench/utils.py
			new_state_dict = OrderedDict()
			for key in state_dict.keys():
				if substr in key:  # to delete prefix 'module.' if it exists
					new_key = key[len(substr):]
					new_state_dict[new_key] = state_dict[key]
				else:
					new_state_dict[key] = state_dict[key]
			return new_state_dict		
			
		if model_name.__contains__("_robustbench"):
			# as long the bug exists in using robustbench downloading: 			
			# https://drive.google.com/drive/u/0/folders/1c3tXIZ_fFcpOkOWF6wHqzAvPtTurbnpR -> gdown
			model = models.resnet50(pretrained=False)
			model_dir = os.path.abspath(os.path.join(self.paths['eval'],"..","models","robustbench"))
			weights = torch.load(os.path.join(model_dir,\
				f"{model_name.split('_robustbench')[0]}.pt"),map_location="cpu")
			weights = rm_substr_from_state_dict(weights['state_dict'],'module.')
			model.load_state_dict(weights)		
			model.__name__ = model_name.lower()
			return model
			
		usermodel = self.models.get(model_name,None)
		
		if usermodel is None:
			assert model_name in __models__, f"Model not available: {model_name}"
			fcn = models.__dict__.get(model_name) # torchvision.models.__dir__()
			if fcn is None:
				raise ValueError(f"model {model_name} not found in torchvision.models")
			else:
				"""
				try:
					if weights is not None:
						model = fcn(weights=weights)
					else:
						raise NotImplementedError("weight definition missing")
				except TypeError:
					model = get_model_pretrained(fcn)
				except NotImplementedError:
				"""
				model = get_model_pretrained(fcn)
				model.__name__ = model_name

				return model
		else:
			#try:
			#	model = models.__dict__.get(usermodel['name'],None)(weights=None)
			#except TypeError:
			model = models.__dict__.get(usermodel['name'],None)(pretrained=False)
			if model is None:
				raise TypeError(f"Model not found: {usermodel['name']}")
			weights = torch.load(usermodel['path'])['model_state_dict']
			weights = rm_substr_from_state_dict(weights,'module.')
			model.load_state_dict(weights)
			model.__name__ = model_name
			return model


	def _get_dataloader(self,path_to_dataset,convert:dict=None):
		"""
		path_to_dataset: dir containing image dataset:  path_to_dataset/folder/*.jpg ...
		convert: input, output dataset names <str>
		"""
		val_transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],\
				std=[0.229, 0.224, 0.225])
			])

		image_folder = datasets.ImageFolder(path_to_dataset)

		if convert:
			image_dataset = ImageNetDatasetConverted(image_folder,\
				val_transform,convert=convert)
		else:
			image_dataset = ImageNetDataset(image_folder,val_transform)


		batch_size = self.batch_size
		# inspired by: https://github.com/BlackHC/toma/blob/master/toma/__init__.py

		isexception=True
		while isexception:
			try:
				self.log(f"current batch_size: {batch_size}")
				dataloader = torch.utils.data.DataLoader(image_dataset,\
					shuffle=False,
					batch_size = batch_size,
					num_workers=self.num_workers,
					pin_memory=self.isgpu)
			except RuntimeError as exception:
				if batch_size > 1 and utils.should_reduce_batch_size(exception):
					batch_size //= 2
					gc.collect()
					if self.isgpu:
						torch.cuda.empty_cache()
				else:
					raise
			else:
				isexception=False

		self.log(f"batch_size finally: {batch_size}")
		return image_dataset,dataloader


	def inference_on_folder(self,model,database:str,split:str=None,mode:str=None,\
			severity:int=None,device='cpu',print_every_percent=25,quiet=False,\
			convert_targets=False):
		"""!

		model: pytorch model (loaded checkpoint)

		with reference to self.databases <dict>:
			database: <str> name of the database
			mode: <str> name of the mode
			sev: <str> name of the severity

		returns: acc@1
		"""
		self.log(f"Inference for {model.__name__} on: {database}/{split}/{mode}/{severity}\n")
		path_to_dataset = self._get_path_from_database(database,split=split,mode=mode,sev=severity)
		image_dataset,dataloader = self._get_dataloader(path_to_dataset)#,convert=convert)
		dataset_size = len(image_dataset)
		if convert_targets:
			converter = ConvertTarget(database,class_to_idx=image_dataset.dataset.class_to_idx,\
				convert=convert_targets)
			self.log(f"loading: {path_to_dataset}, {dataset_size}, {converter}")
		else:
			self.log(f"loading: {path_to_dataset}, {dataset_size} (no cls conversion)")			
		#class_names = image_dataset.classes
		model = model.to(device)
		model.eval()
		since = time.time()
		last = since
		running_corrects = 0
		#self.log(f"[DONE] Initializing inference on {path_to_dataset}")
		with torch.no_grad():
			for i,(inputs, targets) in enumerate(dataloader):
				
				#im = inputs[0]
				#im = im.moveaxis(0,-1)
				#plt.imshow(im)
				#plt.show()
				
				
				inputs = inputs.to(device)
				targets = targets.to(device)
				outputs = model(inputs)
				_, preds = torch.max(outputs, 1)
				if convert_targets:
					targets = converter._get_new_target_batch(targets)

				running_corrects += torch.sum(preds == targets.data)

				if print_every_percent:
					if not i % int(round(len(dataloader)*(print_every_percent)/100)) and i > 0:
						acc = copy.deepcopy(running_corrects).double() / (inputs.size(0)*(i+1))
						now = time.time()
						self.log(f'{100*round(i/len(dataloader),2)}%,'+ \
							f'Acc: {acc:.2f}, batch-time: {(now-last):.4f}s, total: {(now-since):.4f}s.')
						last = now
		del dataloader
		del image_dataset
		now = time.time()
		acc = running_corrects.double() / dataset_size
		self.log(f"Total time: {(now-since):.2f}s. Acc: {acc:.4f}")
		return acc


	def get_savepath(self,model,database,split,mode,severity):
		if split == "val":
			savedir = os.path.join(self.paths["eval"],database,split)
		else:
			savedir = os.path.join(self.paths["eval"],database,split,mode,str(severity))
		savepath = os.path.join(savedir,model.__name__ + ".json")		
		return savepath,savedir


	def report(self,acc,model,database,split=None,mode=None,severity=None):
		"""
		"""
		if split == "val":
			results = {model.__name__:{"acc":float(acc),"database":database,"split":split}}
		else:
			assert mode is not None and severity is not None
			split = 'corruptions' if split is None else split
			results = {model.__name__:{"acc":float(acc),"database":database,"split":split,\
				"mode":mode,"severity":severity}}
		
		
		savepath,savedir = self.get_savepath(model,database,split,mode,severity)

		if not os.path.exists(savedir):
			top = os.path.join(self.paths["eval"],database)
			if not os.path.exists(top):
				self._mirror_subfolder(database)
			top = os.path.join(top,database,split)
			if not os.path.exists(top):
				self._mirror_subfolder(os.path.join(database,split))
			if split != "val":
				top = os.path.join(top,database,split,mode)
				if not os.path.exists(top):
					self._mirror_subfolder(os.path.join(database,split,mode))
				


		with open(savepath,"w") as f:
			json.dump(results,f)
			self.log(f"saved to {savepath}")
		self.log(f"results: {results}")


	def log(self,msg):
		if self.quiet:
			with open(self.paths["log"],'a') as f:
				f.write(str(msg) +"\n")
		else:
			print(msg)


class ImageNetPathsDataset(torch.utils.data.Dataset):
	"""
	ImageNetDataset as a base class , returns images and filepaths
	"""
	def __init__(self,dataset,preprocess):
		self.dataset = dataset
		self.classes = dataset.classes
		self.preprocess = preprocess
		self.index = 0

	def __getitem__(self,index):
		img,__ = self.dataset[index]
		fpath,__ = self.dataset.samples[index]
		self.index = index
		return self.preprocess(img),fpath

	def __next__(self):
		self.index += 1
		return self.__getitem__(self.index)

	def __len__(self):
		return len(self.dataset)


class CreateBenchmark():
	"""

	This generates for a given "clean data folder" --> ImageNette/val
		above folder structure

	should be like:
		images
			ImageNette
				/val

	creates:
		images
			ImageNette
				/val
				/corruptions
					...

	"""
	__seed__ = 0
	__params__ = [3,4,5,6,7,8,9,10]
	__modes__ = {"defocus_spherical":[3,8],\
		"astigmatism":[4,5],
		"coma":[6,7],
		"trefoil":[9,10]
		}

	def __init__(self,testdata_path,psfstack_path=None,all_modes=True,\
			batch_size=50,severities=None):
		"""
		testdata_path: e.g. ..,ImageNette,val
		"""
		print("put all images in a folder data/<dataset_name>/val")
		self.rootdir,__ = os.path.split(testdata_path)
		self.isgpu = torch.cuda.is_available()
		self.device = 'cuda' if self.isgpu else 'cpu'
		self.psf_stack = self._load_blur_kernel_stack(psfstack_path,device=self.device)
		cname = "corruptions"
		corruptions_name = cname+"_rg" if psfstack_path.__contains__("kernel_stack_rg_iccvw.pt") else cname
		self.paths = {"val":testdata_path,\
			"corruptions":os.path.join(self.rootdir,corruptions_name)}

		print(f"Loaded: {psfstack_path}")

		self.create_benchmark_paths()
		self.batch_size = batch_size
		self.severities = severities if severities is not None else [1,2,3,4,5]

		self.val_transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor()#,
			#transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
			])

		# unnormalize
		self.save_transform = transforms.ToPILImage()
		
		dataset = datasets.ImageFolder(self.paths["val"])
		self.dataset = ImageNetPathsDataset(dataset,self.val_transform)
		self.dataloader = torch.utils.data.DataLoader(self.dataset,\
			shuffle=False,
			batch_size = self.batch_size,
			pin_memory=self.isgpu)


	@staticmethod
	def _load_blur_kernel_stack(psfstack_path=None,device='cpu'):
		"""!
		# now:  (<param>,<severity>,<chroma>,<color>,<psfy>,<psfx>)
				if ..ndim==5:
					kernels.unsqueeze(2).shape: ([8, 5, 1, 3, 25, 25])
				else: shape: [8,5,nchroma,3,25,25]
		#.permute(0,1,3,4,2) #,dtype=torch.float32) -- no longer necessary, has the standard shape
		#  kernels.sum(axis=(-1,-2)) not working, so normalize to l1 this way:
		#p,s,c,__,__ = kernels.shape
		"""
		if psfstack_path is None and __name__ == "__main__":
			psfstack_path = os.path.abspath(os.path.join(os.getcwd(),"..","kernel_stack_iccvw.pt"))
		kernels = torch.load(psfstack_path)
		is_gpu = "cuda" if torch.cuda.is_available() else "cpu"
		kernels = kernels.to(device)
		if kernels.ndim == 5:
			kernels = kernels.unsqueeze(2)  # downward compatibility
		print(f"loaded kernel stack of shape: {kernels.shape} ({is_gpu}),\n{psfstack_path}")
		return kernels


	def create_benchmark_paths(self):

		for p in self.__modes__:
			outdir = os.path.join(self.paths['corruptions'],p)
			if not os.path.exists(outdir):
				os.makedirs(outdir)
			self.paths[p] = outdir

		print(f"created folder hierachy in {self.paths['corruptions']}")


	def __get__(self):
		"""

		"""
		for severity in tqdm(self.severities,total=len(self.severities),\
				desc="Severity: ",leave=True):
			for p in tqdm(self.__modes__,desc="Mode: ",leave=False):
				random.seed(self.__seed__,version=2) # every time the same
				params = self.__modes__[p]
				_params = [random.choice(params)-3 for n in range(self.batch_size)]
				outdir = os.path.join(self.paths[p],str(severity))
				if not os.path.exists(outdir):
					os.makedirs(outdir)
				with torch.no_grad():
					for i,(images,fpaths) in enumerate(self.dataloader):
						images = images.to(self.device)
						images = self.__gen__(images,_params,severity=severity).cpu()
						self._create_folders(fpaths,outdir)
						parallel_apply([self._save_image]*len(images),\
							[(im,fpath,outdir) for im,fpath in zip(images,fpaths)])


	def __gen__(self,images,_params,severity=3):
		_batchsize = images.shape[0]
		_chroma = 0
		kernels = self.psf_stack[:, severity-1, _chroma,:]
		return torch.stack(\
			parallel_apply([BlurAugment._apply]*len(images),\
			[(im,kernels[_param,:].unsqueeze(0)) for im,_param in zip(images,_params)]),\
			dim=0).squeeze()


	def _create_folders(self,fpaths,outdir):
		# synset paths for corruption xy
		for fpath in fpaths:
			fpath = os.path.normpath(fpath)
			folder,imname = os.path.split(fpath)
			__,folder = os.path.split(folder)
			outpath = os.path.join(outdir,folder)
			if not os.path.exists(outpath):
				os.makedirs(outpath)


	def _save_image(self,image,fpath,outdir):
		fpath = os.path.normpath(fpath)
		folder,imname = os.path.split(fpath)
		__,folder = os.path.split(folder)
		savepath = os.path.join(outdir,folder,imname)
		transforms.ToPILImage()(torch.clip(image,min=0.0,max=1.0)).save(savepath,quality=85)


def __test_benchmark__(batch_size=10):
	"""
	"""
	testdata_path = os.path.abspath(os.path.join("..","data","images","ImageNette","val"))
	benchmark = CreateBenchmark(testdata_path,batch_size=batch_size)
	benchmark.__get__()

def __test_eval__(ask_dir=True,database=None,**kwargs):

	if not ask_dir:
		path_to_images_folder = os.path.abspath(os.path.join("..","data","images"))
		path_to_root_folder = os.path.abspath(os.path.join("..","data"))
	else:
		path_to_root_folder = get_dir("Get path to root folder (containing eval,images)")
		path_to_images_folder = get_dir("Get path to 'images' folder")
		database = os.path.split(get_dir("Get path to database (top)"))[1]

	evalBench = EvalOpticsBenchmark(path_to_root_folder,path_to_images_folder,
		batch_size=48,num_workers=8)

	model = evalBench._load_model(kwargs['model']) # https://arxiv.org/abs/1602.07360
	print(f"loaded model: {model.__name__}")
	database = 'ImageNette2' if database is None else database
	split = 'val'
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	acc = evalBench.inference_on_folder(model,database,split=split,mode=None,severity=None,\
		device=device)

	evalBench.report(acc,model,database,split=split,mode=None,severity=None)


def get_dir(ttl=""):
	root = tk.Tk()
	root.withdraw()
	return filedialog.askdirectory(title=ttl)


def run_on_all(database="imagenet-1k_val",**kwargs):
	"""!"""
	import warnings
	warnings.filterwarnings("ignore")

	num_workers = kwargs['num_workers']
	batch_size = kwargs['batch_size']
	device = 'cuda' if torch.cuda.is_available() else 'cpu'


	if kwargs.get("path_to_root_folder",None):
		path_to_root_folder = kwargs["path_to_root_folder"]
	else:
		path_to_root_folder = get_dir("Get path to root folder (containing eval,images)")

	path_to_images_folder = os.path.join(path_to_root_folder,"images")
	path_to_user_models = os.path.join(path_to_root_folder,"models")

	if kwargs.get("models",None) == "__imagenet100opticsblur__":
		assert database=="imagenet100"
		path_to_user_models = os.path.join(path_to_user_models,"imagenet100_opticsblur")
	if kwargs.get("models",None) == "__imagenet100__":
		assert database=="imagenet100" or database=="imagenet100-c"
		path_to_user_models = os.path.join(path_to_user_models,"imagenet100")
	if kwargs.get("models",None) == "__imagenet1k__":
		assert database=="imagenet-1k_val"
		path_to_user_models = os.path.join(path_to_user_models,"imagenet1k")


	if database=="imagenet-1k_val" and kwargs.get("models","").__contains__("imagenet100"):
		raise ValueError(\
			f"Combining {database} and models from {kwargs.get('models','')}")

	evalBench = EvalOpticsBenchmark(path_to_root_folder,
		path_to_images_folder,
		path_to_user_models=path_to_user_models,
		batch_size=batch_size,
		num_workers=num_workers,
		quiet=True)

	selections = __registered_model_lists__._get_names()

	model_list = []
	model_list.extend(selections[kwargs.get("models","__add__")]) # default, if not in argparse 

	convert_targets = True if not database=="imagenet-1k_val" else False
	
	if (kwargs.get("models") in ["__imagenet1k__","__imagenette__"]) or \
			kwargs.get("models") in ("__imagenet100__","__imagenet100opticsblur__"):
		convert_targets = False


	print(f"model_list: {model_list}")

	evalBench.log(f"batch_size: {batch_size}, num_workers: {num_workers}, models: {model_list}")
	evalBench.log(f"currently used: {utils.get_gpu_memory()[0]}MB")

	for model_name in tqdm(model_list,desc="Models: ",total=len(model_list),leave=False):
		model = evalBench._load_model(model_name)
		model.cuda()
		evalBench.log(f"currently used (after loading): {utils.get_gpu_memory()[0]}MB")

		splits = ["corruptions"]#['val','corruptions']
		#splits = ['corruptions_rg']

		for split in tqdm(splits,leave=False,total=len(splits),desc="split: "):
			if split == "val":
				modes = [None]
				severities = [None]
			else:
				severities = [1,2,3,4,5]
				
				if database=="imagenet100-c":
					modes = ["brightness","contrast","defocus_blur","gaussian_noise",
						"impulse_noise","shot_noise",'zoom_blur','motion_blur','gaussian_blur',
						'glass_blur','elastic_transform','fog','frost','snow','spatter','speckle_noise',
						'saturate','pixelate','jpeg_compression']
						
				else:
					modes = ["astigmatism","coma","defocus_spherical","trefoil"]#,"defocus_blur"]

			for mode in tqdm(modes,desc="mode: ",total=len(modes),leave=False):
				for severity in tqdm(severities,desc="severity: ",total=len(severities),leave=False):
					
					if os.path.exists(evalBench.get_savepath(model,database,split,mode,severity)[0]):
						if kwargs.get('skip_if_exist',False):
							continue
						else:
							pass
					
					evalBench.log(f"loaded model: {model.__name__}")
					evalBench.log(f"dataset: {database},mode:{mode},severity:{severity}")
					acc = evalBench.inference_on_folder(model,database,split=split,mode=mode,\
						severity=severity,device=device,print_every_percent=None,\
						convert_targets=convert_targets)
					evalBench.report(acc,model,database,split=split,mode=mode,severity=severity)
					gc.collect()
		model.cpu()
		utils.free_memory([model])
		evalBench.log(f"current vram in use (after free): {utils.get_gpu_memory()[0]}MB")
		gc.collect()


def create_benchmark(**kwargs):
	psfstack_path = os.path.abspath(os.path.join(os.getcwd(),"..","kernel_stack"))
	if kwargs.get("use_rg_stack",None):
		psfstack_path += "_rg_iccvw.pt"
	else:
		psfstack_path += "_iccvw.pt"		
		
	testdata_path = kwargs.get("testdata_path") # the path to image val
	batch_size = kwargs.get("batch_size",64)
	benchmark = CreateBenchmark(testdata_path,
		psfstack_path=psfstack_path,\
		batch_size=batch_size,\
		severities=kwargs.get("severities",[1,2,3,4,5]))
	benchmark.__get__()


if __name__ == "__main__":
	#__test__(batch_size=25)
	argparser = argparse.ArgumentParser()
	argparser.add_argument("-t","--testdata_path",default="",type=str)
	argparser.add_argument("--path_to_root_folder",default="",type=str,\
		help="path to root for eval, images etc.")
	argparser.add_argument("-b","--batch_size",default=128,type=int)
	argparser.add_argument("--num_workers",default=6,type=int)
	argparser.add_argument("-m","--model",default="squeezenet1_0",type=str)
	argparser.add_argument("--run_all",action="store_true",\
		help="Select this option to run the benchmark on all available dnns and corruptions")
	argparser.add_argument("--generate_datasets",action="store_true",default=False,\
		help="Select this option to generate image datasets (benchmark) for all corruptions")
	argparser.add_argument("--use_rg_stack",action="store_true",default=False,\
		help="create the OpticsBenchRG ... magenta red and green only")
	argparser.add_argument("--severities",default=[1,2,3,4,5],type=int,nargs='+')
	argparser.add_argument("--database",type=str,default="imagenet-1k_val")
	argparser.add_argument("--models",type=str,default="__a__",\
		help="registered list of models")
	argparser.add_argument("--skip_if_exist",action="store_true",\
		help="skip if *.json exists -- use only if all models will share same data base before/after")
	kwargs = argparser.parse_args().__dict__
	#__test_eval__(**kwargs)

	if kwargs['generate_datasets']:
		assert os.path.exists(kwargs['testdata_path'])
		#if input("Do you want to create the images datasets?") == "yes":
		create_benchmark(**kwargs)


	if kwargs['run_all']:
		print(f"\n\n{'*'*15} This will run eval on all __models__ & all corruptions {'*'*15}\n\n")
		run_on_all(**kwargs)

