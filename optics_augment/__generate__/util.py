# 19.02.2023, Patrick Müller 2023

import sys 
import os 
import shutil
from tqdm import tqdm
import random
import time
import argparse

import torch 


"""

put both repositories in a tree like: 

ImageNet-OpticsBlur
	-- imagenet_optics_benchmark
	-- imagenet_optics_augment

"""


__path_to_benchmark__ = os.path.join(os.path.abspath(\
	os.path.join(os.getcwd(),"..","..")),"imagenet_optics_benchmark")	


def plot_beta_pdf(a,b,n=1000):
    from scipy.stats import beta 
    import matplotlib.pyplot as plt
    label = "alpha: {},beta: {}".format(a,b)
    x = np.linspace(beta.ppf(0.01,a,b),beta.ppf(0.99,a,b),n)
    plt.plot(x,beta.pdf(x,a,b),label=label)
    plt.legend()
    plt.title(label)
    plt.show(block=False)
    input()
    plt.close()


def plot_dirichlet_pdf(a1=0.8,a2=1.2,a3=1.0,a4=1.0,n=100000):
    vals = np.random.dirichlet((a1,a2,a3,a4),size=n)
    alphas = (a1,a2,a3,a4)
    lbl = ["AugMix","OpticsAugment","No augmentation"]
    for i in range(3):
        if i > 1:
            plt.hist(vals[:,i:].mean(axis=1),bins=250)
            m = vals[:,i:].sum(axis=1).mean()
        else:
            plt.hist(vals[:,i],bins=250)
            m = vals[:,i].mean()
        plt.title(f"{lbl[i]}: {alphas[i]}, mean: {m}")
        plt.show(block=False)
        input()
        plt.close()


def get_baselinestack_from_imagenet_optics_benchmark():
	# file: defocus_blur_1.PSF.npy	 --> sev 1...5
	sys.path.insert(1,os.path.join(__path_to_benchmark__,"__generate__"))
	from create_imagenet_optics import load_json, generate_kernel_stack_augment	
	studyname = "base"  
	psfstackpath = os.path.join(__path_to_benchmark__,"current_optics",studyname,\
			f"{studyname}_psf_stack.pt")
	if not os.path.exists(psfstackpath):
		psfstackdir = os.path.split(psfstackpath)[0]
		kernels = [None]*5
		for file in os.listdir(psfstackdir): # has dims [severity,idy,idx] (3d)
			sev = int(os.path.splitext(os.path.splitext(file.split("_")[-1])[0])[0])
			kernels[sev-1] = torch.from_numpy(np.load(os.path.join(psfstackdir,file)))
		kernels = torch.dstack(kernels).moveaxis(-1,0) # (1,5,25,25)
		torch.save(kernels,psfstackpath)
		print(f"saved 'base' kernels to {psfstackpath}")
	return psfstackpath
	

def get_psfstackpath_from_imagenet_optics_benchmark(studyname=None):
	sys.path.insert(1,os.path.join(__path_to_benchmark__,"__generate__"))
	from create_imagenet_optics import get_studyname, load_json, generate_kernel_stack_augment
	if studyname is None:
		studyname = get_studyname()
	psfstackpath = os.path.join(__path_to_benchmark__,"current_optics",studyname,\
			f"{studyname}_psf_stack.pt")
	if not os.path.exists(psfstackpath):
		param_values = load_json(os.path.join(__path_to_benchmark__,\
			"params"),filter_keys=['__chroma_factors__'])
		generate_kernel_stack_augment(param_values=param_values,\
			psfdir=os.path.split(psfstackpath),studyname=studyname)
			
	return psfstackpath


def _get_image_for_plots(torchimage):
	#http://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html#pil_to_tensor
	#print(torchimage.shape)
	return torchimage.squeeze().permute((1,2,0))#torchimage.squeeze().moveaxis(0,-1)



def create_train_val_split_from_train(path_to_train_folder="",path_to_new_dataset="",\
		split=0.95,move_to_val=True,tar=False,**kwargs):
	"""
	should be put in a structure like this (if enough space available):

	imagenet-1k / 
		train
		val 
	imagenet-1k_edit /
		train (will be generated from train)
		val  (will be generated from train)

	args: 
		split (%)   train*split , val*(1-split) 

	
	walk through the directory "train"

	"""			
	assert os.path.exists(path_to_train_folder)
	assert os.path.exists(path_to_new_dataset)
	print(f"selected: (old) {path_to_train_folder}\n(new): {path_to_new_dataset}")
	input("Continue? ")
	# copy folder by folder random filenames
	trainnew = os.path.join(path_to_new_dataset,"train")
	valnew = os.path.join(path_to_new_dataset,"val")

	for f in [path_to_new_dataset,trainnew,valnew]:
		if not os.path.exists(f):
			os.makedirs(f)
			print(f"created: {f}")

	
	# erstelle jetzt aus den Pfaden die gewünschten Splits:
	def ig_f(dir, files):
		return [f for f in files if os.path.isfile(os.path.join(dir, f))]
	def ig_ftypes(dir, files): # ignore_list
		return [f for f in files if f.endswith('.tar') or f.endswith('.zip')]

	if not len(os.listdir(trainnew)):
		shutil.rmtree(trainnew)
		# kopiere zunächst alle Daten in train/
		shutil.copytree(path_to_train_folder,trainnew,ignore=ig_ftypes)
		
	if not len(os.listdir(valnew)):
		# kopiere dateistruktur nach val/ (ohne daten kopieren)
		os.removedirs(valnew)
		shutil.copytree(trainnew,valnew,ignore=ig_f)
	
	if move_to_val:
		# dann verschiebe zufällig samples hieraus in val/
		# imfolder mit 1300 bildern als beispiel: 
		# teile gemäß split auf --> random.choice()
		for folder in tqdm(os.listdir(trainnew),desc="train/val split: ",total=len(os.listdir(trainnew))):
			# for a single folder, do: 	
			# wähle eine anzahl n indizes
			# wähle eine zufällige stichprobe  (ziehen ohne zurücklegen)
			fnames = os.listdir(os.path.join(trainnew,folder))
			sz = len(fnames) # current folder 
			sample = random.sample(fnames,int(round(sz*(1-split))))	
			# now copy all files as selected by the random sampling method: 
			for file in sample:
				__,name = os.path.split(file)
				src = os.path.join(trainnew,folder,name)
				target = os.path.join(valnew,folder,name)
				shutil.move(src,target)
				#print(src)
				#print(target)
				#input()
	
		print(f"[DONE] Creating train / val split from train with {split*100}%")
	
	print("now testing the sizes... / validation.")
	for folder in tqdm(os.listdir(trainnew),desc="(validate)",total=len(os.listdir(trainnew))):
		fnames_train = os.listdir(os.path.join(trainnew,folder))
		fnames_val   = os.listdir(os.path.join(valnew,folder))
		num_train = len(fnames_train)
		num_val = len(fnames_val)
		assert abs(num_val - round((num_train+num_val)*(1-split))) < 2, \
			f"folder: {folder}, mismatch: {num_val}, expected {round((num_train+num_val)*(1-split))} from {num_train}"

	if tar:
		tar_dataset(path_to_new_dataset)


def tar_dataset(path_to_new_dataset=None,tarnow=False,**kwargs):
	"""!"""
	import tarfile 
	# python -m tarfile -c "train/n01440764.tar" "train/n01440764"
	for split in ["train","val"]:
		rootdir = os.path.join(path_to_new_dataset,split)
		for folder in tqdm(os.listdir(rootdir),desc=f"tar: {split}",total=len(os.listdir(rootdir))):
			if os.path.isdir(os.path.join(rootdir,folder)):
				tarname = os.path.join(rootdir,folder + ".tar")
				#print(tarname)
				with tarfile.open(tarname,"w") as t:
					for file in os.listdir(os.path.join(rootdir,folder)):
						t.add(os.path.join(rootdir,folder,file),recursive=True,arcname=file)
				#for file in os.listdir(os.path.join(rootdir,folder)):
	# then tar the whole archives itself, but include only *.tar files:
	if tarnow:
		for split in ["train","val"]:
			rootdir = os.path.join(path_to_new_dataset,split)
			tarname = rootdir + ".tar"
			with tarfile.open(tarname,"w") as t:
				for file in tqdm(os.listdir(rootdir),desc=f"finally *tar the *tars: {split}",\
						total=len(os.listdir(rootdir))):
					if file.endswith('.tar'):
						t.add(os.path.join(rootdir,file),recursive=False,arcname=file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path_to_train_folder",default=None,\
		help="set the path of the imagenet dataset /train folder")
	parser.add_argument("--path_to_new_dataset",default=None,\
		help="set the newpath for the whole directory")
	parser.add_argument("--trainsplit",action="store_true",\
		help="create train / val split from train")
	parser.add_argument("--tar",action="store_true",\
		help="tar the created directory trees")
	parser.add_argument("--split",type=float,default=0.95)
	args = parser.parse_args().__dict__
	
	if args['trainsplit']:
		create_train_val_split_from_train(**args)
	if args['tar']:
		tar_dataset(**args)