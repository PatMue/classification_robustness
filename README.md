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
## Example usage

### Dataset generation (OpticsBench)
```
cd /opticsbench/__generate__
python benchmark.py --generate_datasets --database imagenet-1k_val 
```

### Inference on dataset (OpticsBench)
cd /opticsbench/__generate__
python benchmark.py --run_all --path_to_root_folder <top_level_path_including_images_and_models_folder> --models __all__ 
