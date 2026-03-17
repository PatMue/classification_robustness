# 25.02.2023
import json
import sys 
import os
from tqdm import tqdm
import shutil 
import tkinter as tk 
from tkinter import filedialog
import subprocess as sp
import gc
import torch 
import inspect

from torch.utils.data import Dataset

def get_dir(ttl=""):
	root = tk.Tk()
	root.withdraw()
	return filedialog.askdirectory(title=ttl)


# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
def block_printing(func):
    def func_wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        value = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return value
    return func_wrapper

@block_printing
def get_model_pretrained(fcn):
	return fcn(pretrained=True)


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


class ImageNetDatasetConverted(torch.utils.data.Dataset):
	"""
	convert: <dict> containing input and output dataset names
	"""
	def __init__(self,dataset,preprocess,convert=None):
		self.dataset = dataset
		self.classes = dataset.classes
		self.idx_to_class = {v: k for k,v in self.dataset.class_to_idx.items()}
		self._to_idx = self._class_to_idx(convert) # new dict
		self.preprocess = preprocess

	def _class_to_idx(self,convert):
		return convert_imagenet1k_targets_to_x(**convert)

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
		return convert_imagenet1k_targets_to_x(**convert)

	def _get_new_target(self,target:int):
		cls = self.idx_to_class[target]
		return self._to_idx[cls]

	def _get_new_target_batch(self,targets):
		for i,target in enumerate(targets):
			targets[i] = self._get_new_target(target.item())
		return targets


def _normalize_output(img):
	img = img - img.min()
	img = img / img.max()
	return img
	

def convert_imagenet1k_targets_to_x(input="ImageNet-1k",output="ImageNet-100"):
	"""!
	d = datatsets.ImageFolder(path_to_imagenet1k)
	d.class_to_idx <dict> --> ... {"n101239812": 12, ...  	}
	converts an input dict with 1000 values to a smaller output dict 
	# adds an alternative class_to_idx dict
	"""
	path_to_imagenet_classes = os.path.abspath(os.path.join(os.getcwd(),"..","imagenet_classes.json"))
	with open(path_to_imagenet_classes,"r") as file:
		classes = json.load(file)
	inputs = classes[input]
	outputs = classes[output]
	targets = {k: v for k,v in inputs.items() if k in list(outputs.keys())}
	return targets 


def get_gpu_memory():
	output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
	ACCEPTABLE_AVAILABLE_MEMORY = 1024
	COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
	try:
		memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
	except sp.CalledProcessError as e:
		raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
	memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
	return memory_use_values


def free_memory(to_delete: list, debug=False):
	"""! https://stackoverflow.com/questions/70508960/how-to-free-gpu-memory-in-pytorch """
	calling_namespace = inspect.currentframe().f_back
	for _var in to_delete:
		calling_namespace.f_locals.pop(_var, None)
		gc.collect()
		torch.cuda.empty_cache()


# shamelessly adapted from: https://github.com/BlackHC/toma/blob/10cfe70efaba59ea669c50c0060cfddef65d0b16/toma/torch_cuda_memory.py
def should_reduce_batch_size(exception):
	return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception) 

def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError) and len(exception.args) == 1 and "CUDA out of memory." in exception.args[0]
    )

def is_cudnn_snafu(exception):
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )
# https://github.com/BlackHC/toma/blob/10cfe70efaba59ea669c50c0060cfddef65d0b16/toma/cpu_memory.py
def is_out_of_cpu_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


def get_file(ttl=""):
	root = tk.Tk()
	root.withdraw()
	return filedialog.askopenfilename(title=ttl)
	
def get_dir(ttl=""):
	root = tk.Tk()
	root.withdraw()
	return filedialog.askdirectory(title=ttl)


def create_synsets_imagenet_val(path_to_val="",path_to_val_anns_json=""):
	"""
	
	organize the imagenet-1k validation dataset into classes, save. 
	
	# each file has its own annotation 
	# so read all this  *.xml files and assign each image file to a synset
	"""
	path_to_val_anns_json = get_file(ttl="Get the anns json file for val")
	path_to_val = get_dir(ttl="Get the image directory for 'val' split.")
	
	if not os.path.exists(path_to_val_anns_json):
		path_to_val_anns_xml = get_dir(ttl="Get directory containing xml representation")
		create_json_anns_from_xml_files(path_to_val_anns_xml)		
	
	with open(path_to_val_anns_json,"r") as f:
		anns = json.load(f)['annotations']
	
	impaths = [os.path.join(path_to_val,file) for file in os.listdir(path_to_val) \
		if os.path.isfile(os.path.join(path_to_val,file))]
	# create directory and move files into synset folders: 
	
	for ann,impath in tqdm(zip(anns,impaths),total=len(impaths)):
		__,imname = os.path.split(impath)
		if ann['filename'] in imname:
			folder = ann['name']			
			if not os.path.exists(folder):
				os.makedirs(folder)
			shutil.move(impath,os.path.join(path_to_val,folder,imname))
	print("DONE creating synsets representation from ilsvrc2012_val")
	

def create_json_anns_from_xml_files(path_to_val_anns_xml):
	"""
	https://www.python101.pythonlibrary.org/chapter31_lxml.html
	read in all files and add to dictionary
	
	"""
	files = [os.path.join(path_to_val_anns_xml,file) for file in \
		os.listdir(path_to_val_anns_xml) if file.endswith('.xml')]

	anns = []
	for file in tqdm(files): 
		anns.append(parse_xml(file))

	savepath = os.path.abspath(os.path.join(path_to_val_anns_xml,"..",\
		"ilsvrc2012_val_from_xml.json"))
	anns = {"annotations":anns}
	with open(savepath,"w") as f: 
		json.dump(anns,f,indent=1)
	print(f"saved to {savepath}")
	#print(anns[0:10])


def parse_xml(xml_file):	
	from lxml import etree
	with open(xml_file,'r') as f:
		xml = f.read()
	root = etree.fromstring(xml)
	d = {}
	for elem in root.getchildren():
		if elem.tag == "filename":
			d[elem.tag] = elem.text
		elif elem.tag == "object":
			for e in elem.getchildren():
				if e.tag == "name":
					d[e.tag] = e.text
	return d
	
if __name__ == "__main__":
	#path_to_val_anns_xml = os.path.abspath(os.path.join("..","data","val"))
	create_synsets_imagenet_val()