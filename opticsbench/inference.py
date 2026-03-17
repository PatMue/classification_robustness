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
__author__ = "Patrick M\"uller"

import os
import shutil
import sys
import argparse
import time
import copy
from tqdm import tqdm
import json
import datetime
import logging
import gc
from pathlib import Path

import torch
import torchvision
from torchvision import datasets,transforms,models
from PIL import Image

#from robustbench.model_zoo.imagenet import imagenet_models  # switched to manual download since there are bugs in current version of robustbench repo 
import utils 
from utils import block_printing, get_model_pretrained, ImageNetDatasetConverted, ConvertTarget, ImageNetDataset
import __registered_model_lists__

logger = logging.getLogger(__name__)

try:
	__models__ = torchvision.models.list_models()
except AttributeError:
	__models__ = [m for m in dir(models) if not m.startswith("_") and m.islower()]

"""
1. Assign paths
2. Run inference
3. save results
4. report results

# create dataloader for each folder:

data/
 images/
  ImageNet-100
    /val
	/corruptions
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
eval/
  ImageNet-100
	/val
	/corruptions
    	astigmatism
			resnet50
			efficientnet_b0
			convnext_xy
			...
	 	coma
			resnet50
			efficientnet_b0
			convnext_xy
			...
		...
		common2d
			defocus_blur
				resnet50
				efficientnet_b0
				convnext_xy
				...
		...
  ImageNet-1k
	/val
	/corruptions
	...

models/ 
	...

# 	evaluate image folder in images/ for one epoch,
	save results to same hierarchy: eval/
"""


class EvalOpticsBenchmark():

	def __init__(self,path_to_root,path_to_images_folder,path_to_user_models="",\
			batch_size=25,num_workers=4,quiet=False):
		"""
		take a neural network, infer on dataset, save results to mirrored folder
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
		self.paths = {
			"eval":Path(path_to_root) / "eval",
			"images":Path(path_to_images_folder),\
			"models":Path(path_to_user_models) if path_to_user_models else None,
			"log":Path(path_to_root) / "eval" / "logfile.log"
			}
		self.isgpu = torch.cuda.is_available()
		self.batch_size = batch_size
		self.num_workers = num_workers

		if not self.paths["eval"].exists():
			self._mirror_folders()

		if quiet:
			self._get_logfile()
		self.quiet = quiet

		self.databases 	= self._get_available_datasets()
		self.models 	= self._get_available_user_models()
		self.log(f"databases: {self.databases}")
		self.log(f"models: {self.models}")



	def _get_logfile(self):
		"""Run once per class initialization, saved to eval folder top level"""
		self.paths["eval"].mkdir(parents=True, exist_ok=True)
		i = 0
		log_path = self.paths["log"]
		while log_path.exists():
			log_path = self.paths["eval"] / f"logfile_{i}.log"
			i += 1
		self.paths["log"] = log_path
		with open(self.paths['log'], "w") as f:
			f.write(f"This is the log file for evaloptics on {datetime.datetime.now()}\n")


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


	def _get_available_datasets(self):
		"""
		returns a <dict> containing all available datasets
		databases[dataset_name][split][mode][severity] = path_to_images
		"""
		databases = {}
		for db_dir in [d for d in self.paths["images"].iterdir() if d.is_dir()]:
			db_name = db_dir.name
			databases[db_name] = {"path": db_dir, "subsets": {}}
			
			for split_dir in [s for s in db_dir.iterdir() if s.is_dir()]:
				split_name = split_dir.name
				
				if split_name in ["corruptions", "corruptions_rg"]:
					databases[db_name]["subsets"][split_name] = {}
					for mode_dir in [m for m in split_dir.iterdir() if m.is_dir()]:
						mode_name = mode_dir.name
						databases[db_name]["subsets"][split_name][mode_name] = {}
						for sev_dir in [sv for sv in mode_dir.iterdir() if sv.is_dir()]:
							databases[db_name]["subsets"][split_name][mode_name][sev_dir.name] = sev_dir
				elif split_name == "val":
					databases[db_name]["subsets"][split_name] = split_dir
					
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
			return str(database["subsets"][split])

		self.log(f"split: {split}, {mode}, {sev}")
		assert mode is not None, f"requires mode - corruption, but is: {mode}"
		assert sev in [1,2,3,4,5], f"mode requires 'sev', but is: {sev},{type(sev)}"
		# now it is in the keys
		try:
			return str(database["subsets"][split][mode][str(sev)])
		except KeyError:
			return str(database["subsets"][split][mode][sev])


	def _get_available_user_models(self):
		"""
		get list of user trained models (requires torchvision eqivalent description)
		"""
		if not self.paths["models"] or not self.paths["models"].exists():
			return {}
		
		def get_name_from_savepath(p):
			# prerequisite: needs to be in torchvision.models listed
			# find name in __models__
			name =  [model for model in __models__ if model in p]
			if len(name) == 1:
				return name[0]
			raise ValueError(f"Model not found in __models__: {p}")

		if self.paths['models'].exists():
			user_models = {}
			for p in self.paths["models"].iterdir():
				if p.suffix =='.pt':
					user_models[p.with_suffix('')] = {"path": \
						os.path.join(self.paths["models"] / p), "name":get_name_from_savepath(str(p))}
			return user_models
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
				model = get_model_pretrained(fcn)
				model.__name__ = model_name

				return model
		else:
			model = models.__dict__.get(usermodel['name'],None)(weights=None)
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
		with torch.no_grad():
			for i,(inputs, targets) in enumerate(dataloader):
								
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
			savedir = self.paths["eval"] / database / split
		else:
			savedir = self.paths["eval"] / database / split / mode / str(severity)
		savepath = savedir / (model.__name__ + ".json")		
		return str(savepath), str(savedir)


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

		Path(savedir).mkdir(parents=True, exist_ok=True)
				
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


def run_on_all(database="imagenet-1k_val",**kwargs):
	"""!"""
	import warnings
	warnings.filterwarnings("ignore")

	num_workers = kwargs['num_workers']
	batch_size = kwargs['batch_size']
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	path_to_root_folder = kwargs["path_to_root_folder"]
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

		splits = ["val","corruptions"]#['val','corruptions']
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
		del model
		gc.collect()
		torch.cuda.empty_cache()
		evalBench.log(f"current vram in use (after free): {utils.get_gpu_memory()[0]}MB")


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	
	argparser.add_argument("--path_to_root_folder",default="",type=str,help="path to root for eval, images etc.")
	argparser.add_argument("--database",type=str,default="imagenet-1k_val")

	argparser.add_argument("--skip_if_exist",action="store_true",\
		help="skip if *.json exists. use only if all models will share same data base before/after")
	argparser.add_argument("-m","--model",default="squeezenet1_0",type=str)
	argparser.add_argument("--models",type=str,default="__all__",help="registered list of models")
	argparser.add_argument("-b","--batch_size",default=128,type=int)
	argparser.add_argument("--num_workers",default=6,type=int)
	


	kwargs = argparser.parse_args().__dict__

	print(f"\n\n{'*'*15} This will run eval on all __models__ & all corruptions {'*'*15}\n\n")
	run_on_all(**kwargs)

