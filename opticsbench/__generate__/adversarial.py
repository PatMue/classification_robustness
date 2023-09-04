# # (c) Patrick Müller 2023 

# 08.03.2023
# !pip install -q git+https://github.com/fra31/auto-attack
# import warnings
# warnings.filterwarnings('ignore')
import argparse
import os
import gc
import math

import autoattack
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models as models


__architectures__ = ["densenet161","resnext50_32x4d","efficientnet_b0","resnet101",\
	"mobilenet_v3_large","efficientnet_b0"]


# from https://github.com/RobustBench/robustbench/blob/master/robustbench/utils.py
def clean_accuracy(model,
		x: torch.Tensor,
		y: torch.Tensor,
		batch_size: int = 100,
		device: torch.device = None):
	if device is None:
		device = x.device
	acc = 0.
	n_batches = math.ceil(x.shape[0] / batch_size)
	with torch.no_grad():
		for counter in range(n_batches):
			x_curr = x[counter * batch_size:(counter + 1) *
					   batch_size].to(device)
			y_curr = y[counter * batch_size:(counter + 1) *
					   batch_size].to(device)

			output = model(x_curr)
			acc += (output.max(1)[1] == y_curr).float().sum()

	return acc.item() / x.shape[0]


def load_model(model_dir,model_name,mean=None,std=None):
	# get the torch equivalent: 
	model_path = os.path.join(model_dir,model_name)
	pt_name = [m for m in __architectures__ if m in model_name][0]
	model = models.__dict__[pt_name](pretrained=False)
	model.load_state_dict(torch.load(model_path,map_location="cpu")["model_state_dict"])
	model.train()
	#model = robustbench.model_zoo.architectures.utils_architectures.normalize_model(model,mean,std)
	model.__name__ = model_name
	return model


def main(args):
	
	norm='L2'  # 'Linf'
	eps = 4/255 #8/255
	version = 'rand'
	n_restarts = 5
	
	root_dir = args.root_dir
	batch_size = args.batch_size
	num_workers = args.num_workers
	
	data_dir = os.path.join(root_dir,"images","imagenet100-adv","test","val")
	model_dir = os.path.join(root_dir,"models","imagenet100")
	log_dir = os.path.join(root_dir,"eval","autoattack_logs")
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	__all__ = [m for m in os.listdir(model_dir) if m.endswith('.pt')]
	print(f"all models: {__all__}")
	
	if args.model_name in __all__:
		model_name = args.model_name 
	else:
		print(__all__)
		raise(ValueError(f"{args.model_name}"))

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	isgpu = torch.cuda.is_available()

	exists = True
	n = 0
	while exists:
		log_path = os.path.join(log_dir,f"stats_{model_name}_{n}.json")
		if os.path.exists(log_path):
			n+=1
		else:
			exists = False
			
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)

	transforms_test = transforms.Compose(\
		[transforms.Resize(256),transforms.CenterCrop(224),\
		transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])

	imagenet100 = datasets.ImageFolder(data_dir, transforms_test)
	
	test_loader = torch.utils.data.DataLoader(imagenet100,
								  batch_size=batch_size,
								  shuffle=False,
								  num_workers=num_workers,\
								  pin_memory=isgpu)

	
	torch.manual_seed(0) # reproducibility of the test set, although now shuffle=False 
	model = load_model(model_dir,model_name,mean=mean,std=std)
	model.train()
	model.to(device)
	
	for i,(images,targets) in enumerate(test_loader):
		images = images.to(device)
		targets = targets.to(device)
		acc_adv = attack_model(model,images,targets,device=device,norm=norm,\
			eps=eps,version=version,log_path=log_path,bs=batch_size,\
			n_restarts=n_restarts)
		
		print(f"[DONE] {i}/{len(test_loader)}, acc_adv: {acc_adv}")
	
	print(f"attacked: {model.__name__}")
	print("\n")
	model.to('cpu')
	del model 
	gc.collect()


def attack_model(model,images,targets,device='cpu',norm='L2',eps=4/255,version='rand',
		log_path=None,seed=None,verbose=True,bs=250,n_restarts=5):
	# # attacks_to_run=['apgd-ce', 'apgd-dlr'])
	
	print(f"\nAttacking: {model.__name__}\n")
	adversary = autoattack.AutoAttack(model, \
		norm=norm, eps=eps, version=version,
		device=device,\
		log_path=log_path,\
		seed=seed)
		
	adversary.apgd.n_restarts = n_restarts
	images_adv = adversary.run_standard_evaluation(images,\
		targets,bs=bs)
		
	acc = clean_accuracy(model,images_adv,targets,batch_size=bs,device=device)
	return acc
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name",type=str,default=None)
	parser.add_argument("--root_dir",type=str,default=None)
	parser.add_argument("--batch_size",type=int,default=32)
	parser.add_argument("--num_workers",type=int,default=4)
	
	args = parser.parse_args()
	main(args)