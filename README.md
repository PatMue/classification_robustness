# Classification robustness to common optical aberrations
(c) Patrick Müller 2023, licensed under MIT-license

AROW Workshop ICCV2023 Code submission. Classification robustness to common optical aberrations.


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




## Example usage (OpticsBench)

Creates / requires a folder hierarchy like: 

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


### Dataset generation
```
cd /opticsbench/__generate__
python benchmark.py --generate_datasets --database imagenet-1k_val 
```

### Inference / Evaluate
```
cd /opticsbench/__generate__
python benchmark.py --run_all --path_to_root_folder <root> --models __all__ 
```
Available model lists can be found in opticsbench/__generate__/__registered_model_lists.py__
