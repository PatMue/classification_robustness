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


