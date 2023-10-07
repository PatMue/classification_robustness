# Classification robustness to common optical aberrations
(c) Patrick Müller 2023 (for license see /opticsbench, /opticsaugment)

[ICCV2023 AROW Workshop official code submission. Classification robustness to common optical aberrations](https://openaccess.thecvf.com/content/ICCV2023W/AROW/html/Muller_Classification_Robustness_to_Common_Optical_Aberrations_ICCVW_2023_paper.html)

#### Why?
Computer vision using deep neural networks (DNNs) has brought about seminal changes in people's lives. DNNs have to behave in a robust way to disturbances such as noise, pixelation, or blur. Blur directly impacts the performance of DNNs, which are often approximated as a disk-shaped kernel to model defocus. However, optics suggests that there are different kernel shapes depending on wavelength and location caused by optical aberrations. In practice, as the optical quality of a lens decreases, such aberrations increase. 

#### What?
* We propose OpticsBench, a benchmark for investigating robustness to realistic, practically relevant optical blur effects. Each corruption represents an optical aberration (coma, astigmatism, spherical, trefoil) derived from Zernike Polynomials. Experiments on ImageNet show that for a variety of different pre-trained DNNs, the performance varies strongly compared to disk-shaped kernels, indicating the necessity of considering realistic image degradations.
* We propose OpticsAugment, an augmentation method, that increases robustness by using optical kernels. Compared to a conventionally trained ResNeXt50, training with OpticsAugment achieves an average performance gain of 21.7% points on OpticsBench and 6.8% points on 2D common corruptions.

#### Code

* `/opticsbench` contains the python code to both create our blur corruption datasets and evaluate pytorch DNNs on this data. User models can be registered in /models
* `/opticsaugment` contains the python code to train models with OpticsAugment. OpticsAugment itself is defined in `/opticsaugment/__generate__/recipes/_augment.py`.

## Citation
If you find this useful in your research, please consider citing:

```
@InProceedings{Muller_2023_ICCV,
    author    = {M\"uller, Patrick and Braun, Alexander and Keuper, Margret},
    title     = {Classification Robustness to Common Optical Aberrations},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {3632-3643}
}
```

## Leaderboard (OpticsBench ImageNet-100)

Accuracies w/wo OpticsAugment evaluated on ImageNet-100 OpticsBench. Average over all corruptions. 

Model | 1 | 2 | 3 | 4 | 5
--- | --- | --- | --- |--- |--- 
DenseNet __(ours)__ | __68.22__ | __65.33__ | __56.33__ | __41.60__ | __30.13__
DenseNet | 53.45 | 43.37 | 29.07 | 20.62 | 16.30
EfficientNet __(ours)__ | __61.00__ | __55.34__ | __42.14__ | __30.27__ | __23.35__ 
EfficientNet | 52.55 | 42.74 | 29.24 | 20.84 | 16.00
MobileNet __(ours)__ | __57.59__ | __52.30__ | __38.58__ | __27.51__ | __20.54__ 
MobileNet | 49.47 | 39.57 | 24.78 | 17.42 | 13.27
ResNet101 __(ours)__ | __69.90__ | __67.68__ | __61.36__ | __49.04__ | __37.80__ 
ResNet101 | 59.92 | 51.44 | 40.21 | 31.65 | 25.73
ResNeXt50 __(ours)__ | __65.14__ | __62.68__ | __54.44__ | __39.90__ | __28.45__ 
ResNeXt50 | 47.74 | 38.19 | 24.88 | 17.58 | 13.69

We highly encourage researchers to modify the pre-defined hyperparameters and report results on OpticsBench ImageNet-1k or ImageNet-100. If you want to be included in the leaderboard, feel free to contact patrick.mueller@student.uni-siegen.de.

### Common corruptions
Performance *gain* with OpticsAugment on all 2D common corruptions as average difference in accuracy across all corruptions in %-points for each severity.

DNN | 1 | 2 | 3 | 4 | 5 
--- | --- | --- | --- |--- |--- 
 DenseNet161  | 5.08 |  7.55 | 8.73  | 7.30 | 5.38 
 ResNeXt50 | 5.11 |  7.63 |  8.68 | 7.18 |  5.27
 ResNet101 | 1.25 | 3.07 | 4.55 | 4.90 | 4.10
 MobileNet | 3.58 | 4.92 | 4.78 | 3.69 | 3.07 
 EfficientNet  | 4.35 | 6.32 | 6.70 | 4.62 | 3.69

## Example usage (OpticsAugment)
Train a DNN (architecture available in pytorch) using OpticsAugment augmentation. We define basic recipes for different DNNs in `optics_augment/__generate__/recipes/`. We highly encourage researchers to modify the pre-defined hyperparameters and report results on OpticsBench.
```
cd /optics_augment/__generate__
```
```
python train_dnn.py --root_dir <path_to_dataset> --model_dir $path_to_modeldir --name <model_name> --num_workers <num_workers>
```

Pipelining with AugMix is also possible by adding `--augmix`.


## Example usage (OpticsBench)

### Dataset generation
```
cd /opticsbench/__generate__
```
```
python benchmark.py --generate_datasets --database imagenet-1k_val 
```
Creates folders: `data/images/<dataset>/<val,corruptions>/<corruption_name>/<severity>/`

### Inference / Evaluate
```
cd /opticsbench/__generate__
```
```
python benchmark.py --run_all --path_to_root_folder <root> --models __all__ 
```

* The folder hierarchy of the image dataset is mirrored to the eval folder at initialization of the inference / evaluation.
* Creates elements: `data/eval/<dataset>/<val,corruptions>/<corruption_name>/<severity>/<model_name>.json`
* Each json contains information regarding severity, corruption, dataset and the evaluated metrics (accuracy)

### Adding user models

Available model lists can be found in `opticsbench/__generate__/__registered_model_lists.py__`. To add user models do the following:

* Put a pytorch checkpoint into `/data/models/*.pt`
* Then register by adding the name in `opticsbench/__generate__/__registered_model_lists.py`
* and call e.g. as `python benchmark.py --run_all --path_to_root_folder <root> --models <user_model_list_name> `

### Example tree (folder / files hierarchy): 

```
root/
	images
		ImageNette
			/val
			/corruptions
				opticsblur
						astigmatism
							1
							2
							3
							4
							5
						coma
							1
							2
							...
						trefoil
							...
						defocus_spherical
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
						astigmatism
                            1/
							resnet50.json
							efficientnet_b0.json
							convnext_xy.json
							...
							resnet50.json
							...
                            2/
                            3/...

						coma
                            1/
							resnet50.json
							efficientnet_b0.json
							convnext_xy.json
							...
						trefoil
                            1/
							resnet50.json
							efficientnet_b0.json
							convnext_xy.json
							...
						defocus_spherical
                            1/
							resnet50.json
							efficientnet_b0.json
							convnext_xy.json
							...
				common2d
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
		resnet50_augmix_optics_augment
		efficientnet_b0_augmix_optics_augment
		convnext_xy_augmix
		...
	...
```


