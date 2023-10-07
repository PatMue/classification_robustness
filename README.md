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
data/
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
							resnet50_sev1.json
							efficientnet_b0_sev1.json
							convnext_xy_sev1.json
							...
							resnet50_sev2.json
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
		resnet50_augmix_blur_augment
		efficientnet_b0_augmix_blur_augment
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
python benchmark.py --run_all --path_to_root_folder <path_including_images_and_models_folder> --models __all__ 
```
