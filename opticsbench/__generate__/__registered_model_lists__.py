#model_list = __models__[__models__.index('resnet101'):]


def _get_names():

	__all__ = ['regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf','vit_b_16','vit_b_32','vit_l_16','vit_l_32','convnext_tiny','convnext_small','convnext_base','convnext_large','alexnet', 'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf','resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'inception_v3', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'mnasnet0_5', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'squeezenet1_0', 'squeezenet1_1']


	__half1__ = ['regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf','vit_b_16','vit_b_32','vit_l_16','vit_l_32','convnext_tiny','convnext_small','convnext_base','convnext_large','alexnet', 'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf','resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152','resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'inception_v3']
	__half2__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'mnasnet0_5', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'squeezenet1_0', 'squeezenet1_1']

	__sim__ = ['vit_b_16','vit_b_32','vit_l_16','vit_l_32','convnext_tiny','convnext_small',
		'convnext_base','convnext_large'] 


	__a__ = ['alexnet', 'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf','resnet18',
		'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
		'wide_resnet50_2','wide_resnet101_2', 'inception_v3']


	__b__ = ['densenet121', 'densenet169','mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
		'efficientnet_b0','efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 
		'efficientnet_b5','efficientnet_b6', 'efficientnet_b7', 'mnasnet0_5', 'regnet_x_400mf', 
		'regnet_x_800mf','regnet_x_1_6gf', 'mnasnet0_75']


	__c__ = ['densenet201', 'densenet161','mnasnet1_0', 'mnasnet1_3', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
		'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
		'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'squeezenet1_0', 'squeezenet1_1']


	__robustbench__ = ['Hendrycks2020Many','Hendrycks2020AugMix','Geirhos2018_SIN_IN',\
		'Geirhos2018_SIN_IN_IN','Erichson2022NoisyMix','Salman2020Do_R50']		
	__robustbench__ = [r + "_robustbench" for r in __robustbench__] # append for internal use


	__imagenet1k__ = ['efficientnet_b0_sgd_opticsaugment'] #,'mobilenet_v3_large_kuan_wang_warm_restarts_opticsaugment']
	

	__imagenet100opticsblur__ = ['alexnet_sgd', 'efficientnet_b0_sgd', 'efficientnet_b4_sgd',
		'mobilenet_v3_large_kuan_wang_warm_restarts_temp', 'resnet101_sgd', 'resnet50_rmsprop',
		'resnet50_sgd','resnext50_32x4d_sgd', 'vgg16_sgd_temp']  # deprecated (share parts of test=validation set)


	__imagenet100__ =  ['densenet161_sgd_opticsaugment','densenet161_sgd_clean',\
		'efficientnet_b0_sgd_clean','efficientnet_b0_sgd_opticsaugment',
		"resnet101_sgd_opticsaugment","resnet101_sgd_clean",
		'mobilenet_v3_large_kuan_wang_warm_restarts_opticsaugment','mobilenet_v3_large_kuan_wang_warm_restarts_clean',
		'resnext50_32x4d_sgd_opticsaugment','resnext50_32x4d_sgd_clean',\
		'efficientnet_b0_sgd_augmix_opticsaugment','mobilenet_v3_large_kuan_wang_warm_restarts_augmix_opticsaugment']


	selections = {"__sim__":__sim__,"__a__":__a__,"__b__":__b__,"__c__":__c__,\
		"__all__":__all__,\
		"__half1__":__half1__,
		"__half2__":__half2__,
		"__robustbench__":__robustbench__,\
		"__imagenet1k__":__imagenet1k__,\
		"__imagenet100opticsblur__":__imagenet100opticsblur__,
		"__imagenet100__":__imagenet100__
		}
	
	return selections 
		
