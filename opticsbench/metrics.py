# 24.02.2023
# maybe later included in zom.Metrics.nn_metrics

# may require imports from  ../imagenet_optics_augment
import os 
import copy 

import torch 
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def __test__(batch_size=10):
	from torchvision import models,datasets
	
	isgpu = torch.cuda.is_available()
	
	if isgpu:
		model = models.mobilenet_v3_large(pretrained=True)
	else:
		model = models.squeezenet1_1(pretrained=True)

	valdir = os.path.abspath(os.path.join("","..","..","..","..",'ImageNette','val'))
	
	print("Loading the model: Done")

	# dataloader on ImageNette/val  ~ 4k images
	preprocess = transforms.Compose([\
			transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
	dataset = datasets.ImageFolder(valdir,transform=preprocess)
	dataloader = torch.utils.data.DataLoader(dataset,
		batch_size=batch_size,
		shuffle=True, # just to have some more diverse data... 
		sampler=None,
		pin_memory=isgpu)
		
	print("DONE: Preprocessing and data loader initialization")

	for i,(batch,targets) in enumerate(dataloader):
		print(f"finished dataloading for batch: {i}")
		#show_input_data(batch,targets,dataset.classes)
		saliency = get_saliency_map(batch,targets,model)
		print(f"finished computing saliency for batch: {i}")
		show_saliency_maps(saliency,batch,targets,dataset.classes)
		if input("Stop? "):
			break


def get_saliency_map(batch,targets,model):
	"""!
	
	compute a saliency map for image x and network "model" with loaded weights (supervised learning)
	
	modified from:
	https://github.com/sijoonlee/deep_learning/blob/master/cs231n/NetworkVisualization-PyTorch.ipynb
	https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad_.html	
	
	args: 
		batch:  img batch of shape (N,3,H,W)
		targets: targets for the batch; (N,)
		model: pretrained dnn, pytorch 
	"""
	model.eval()  # do not train
	batch = torch.FloatTensor(batch)#,requires_grad=True)
	batch.requires_grad_() # to track the gradient 
	saliency = None 	
	scores = model(batch)
	scores = (scores.gather(1,targets.view(-1,1)).squeeze())
	scores.backward(torch.FloatTensor([1.0]*scores.shape[0]))
	saliency,__ = torch.max(batch.grad.data.abs(),dim=1)
	return saliency


def show_input_data(batch,targets,classes):
	"""
	visualize the obtained saliency maps for image batch	
	targets, classes
	"""	
	batch_size = batch.shape[0]
	for i in range(batch_size):
		plt.subplot(2,batch_size,i+1)
		plt.imshow(batch[i].moveaxis(0,-1)/batch[i].max())
		plt.axis('off')
		plt.title(classes[targets[i]])
		plt.gcf().set_size_inches(12,2)
	plt.show(block=False)
	input()
	plt.close()



def show_saliency_maps(saliency,batch,targets,classes,cmap="hot"):
	"""
	visualize the obtained saliency maps for image batch	
	targets, classes
	"""		
	batch.requires_grad_(requires_grad=False)	
	batch = torch.stack([normalize_output(b) for b in batch])
	batch_size = batch.shape[0]
	#p = transforms.Compose([\
	#	transforms.Normalize(mean=[0,0,0],\
	#		std=(1.0/torch.Tensor([-0.229, -0.224, -0.225])).tolist())#\
	#	#transforms.Normalize(mean=[-0.485, -0.456, -0.406],std =[1,1,1])
	#	])
	#batch = p(batch)
	for i in range(batch_size):
		plt.subplot(2,batch_size,i+1)
		plt.imshow(batch[i].moveaxis(0,-1)/batch[i].max())
		plt.axis('off')
		plt.title(classes[targets[i]])
		plt.subplot(2,batch_size,batch_size + i +1)
		plt.imshow(saliency[i],cmap=cmap)
		plt.axis('off')
		plt.gcf().set_size_inches(12,8)
	plt.gcf().tight_layout()
	plt.show(block=False)
	input()
	plt.close()

def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img

def get_acc_k():
	"""!
	
	# basically  1 - torch.topk(...,k=k)
	
	"""
	raise(NotImplementedError)
	
	

class VisFeatureMap():
	
	"""
	see https://discuss.pytorch.org/t/visualize-feature-map/29597/2
	"""
	
	def __init__(self,model):
		self.model = copy.deepcopy(model)
		self.activation = {}
	
	def _feed(self,batch):
		"""
		"""
		out = self.model(batch)
		

	def get_activation(self,name):
		def hook(model, input, output):
			activation[name] = output.detach()
		return hook	



	
def get_disparity_map():
	"""!
	
	as in Vasiljevic et al. 2017
	
	"""
	raise(NotImplementedError)
	
	
if __name__ == "__main__":
	__test__()