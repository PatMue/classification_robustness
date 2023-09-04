# Patrick Müller (c) 2022, 09.12.2022
import os

from . import _basic_training


def get_modified_training_setup(setup,**kwargs):
    """
    http://caffe.berkeleyvision.org/tutorial/solver.html
    
    """
    raise NotImplementedError
    batch_size = 32
    num_epochs = 90
    learning_rate = 0.04
    momentum =0.9
    weight_decay = 0.0002# l2 weight decay for RMSprop, https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    
    # linear lr scheduler: 
    start_factor = None
    total_iters = 16 
    max_iters = 170000

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_dnn.parameters(), lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    # Decay LR linearly: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
    selected_lr_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=start_factor,\
        total_iters=total_iters,end_factor=1.0,last_epoch=-1)

    training_setup = {}
    training_setup["num_workers"] = kwargs.get("num_workers",os.cpu_count()//2)
    training_setup["num_epochs"] = kwargs.get("num_epochs",num_epochs)
    training_setup["batch_size"] = kwargs.get("batch_size",batch_size)
    training_setup["criterion"] = criterion
    training_setup["optimizer"] = optimizer
    training_setup["scheduler"] = selected_lr_scheduler


def load_recipe(model_dnn,setup=None,**kwargs):
    """
    https://github.com/pytorch/vision/tree/main/references/classification#Resnet
    
	https://github.com/DeepScale/SqueezeNet  , version 1.0
	https://github.com/songhan/SqueezeNet-DSD-Training
	https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/solver.prototxt
	
	test_iter: 2000 #not subject to iter_size
	test_interval: 1000
	base_lr: 0.04
	max_iter: 170000
	iter_size: 16 #global batch size = batch_size * iter_size
	lr_policy: "poly"
	power: 1.0 #linearly decrease LR
	momentum: 0.9
	weight_decay: 0.0002
	random_seed: 42
	
	# linearly decrease learning rate  LinearLR
	
	pytorch (Collobert et al., 2011)
    """
    
    setup["training"] = get_modified_training_setup(model_dnn,mode="sgd")
    
    name = model_dnn.__name__ + "_sgd"
    return setup,model_dnn,name
