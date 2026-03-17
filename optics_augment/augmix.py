# modified from: https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/augmentations.py
# originally used in AugMix, Hendrycks et al.

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
IMAGE_SIZE = 224

# modified from https://github.com/google-research/augmix   (Apache 2.0 license)
# https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/imagenet.py


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k.
	adapted from: https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/imagenet.py
	"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def aug(image, preprocess, severity=3, all_ops=False,\
		aug_prob_coeff=1.0,mixture_depth=-1,mixture_width=3):
	"""Perform AugMix augmentations and compute mixture.
	Args:
	image: PIL.Image input image
	preprocess: Preprocessing function which should return a torch tensor.
	Returns:
	mixed: Augmented and mixed image.

	modified from: https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/imagenet.py
	"""
	aug_list = augmentations
	if all_ops:
		aug_list = augmentations_all

	ws = np.float32(
	  np.random.dirichlet([aug_prob_coeff] * mixture_width))
	m = np.float32(np.random.beta(aug_prob_coeff, aug_prob_coeff))

	mix = torch.zeros_like(preprocess(image))
	for i in range(mixture_width):
		image_aug = image.copy()
		depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
		for _ in range(depth):
			op = np.random.choice(aug_list)
			image_aug = op(image_aug, severity)
			# Preprocessing commutes since all coefficients are convex
			mix += ws[i] * preprocess(image_aug)

	mixed = (1 - m) * preprocess(image) + m * mix
	return mixed


def aug_pipelined(image,m, preprocess, severity=3,\
		aug_prob_coeff=1.0,mixture_depth=-1,mixture_width=3):
	"""Perform AugMix augmentations and compute mixture.
	modified from: https://github.com/google-research/augmix/blob/9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4/imagenet.py
	Args:
	image: PIL.Image input image
	preprocess: Preprocessing function which should return a torch tensor.
	Returns:
	mixed: Augmented and mixed image.
	
	* modification: accept m as variable
	
	"""
	aug_list = augmentations

	ws = np.float32(np.random.dirichlet([aug_prob_coeff] * mixture_width))
	mix = torch.zeros_like(preprocess(image))
	for i in range(mixture_width):
		image_aug = image.copy()
		depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
		for _ in range(depth):
			op = np.random.choice(aug_list)
			image_aug = op(image_aug, severity)
			mix += ws[i] * preprocess(image_aug)

	mixed = (1 - m) * preprocess(image) + m * mix
	return mixed


class AugMixDataset(torch.utils.data.Dataset):
	"""Dataset wrapper to perform AugMix augmentation.
	
	* modified to accept variable: <bool> pipelined 	
	* __getitem__ method computes from dirichlet distribution and returns random number "prob2"
	"""

	def __init__(self, dataset, preprocess,pipelined=False,alpha=1.0):
		self.dataset = dataset #  [(PIL Image, target), ...]
		self.classes = dataset.classes
		self.targets = dataset.targets # a list
		self.preprocess = preprocess
		self.pipelined = pipelined
		self.alpha = alpha 

	def __getitem__(self, i):
		x, y = self.dataset[i]
		if self.pipelined:
			prob1 , prob2 = np.float32(np.random.dirichlet([self.alpha]*4))[:2]
			return (aug_pipelined(x,prob1,self.preprocess),prob2), y 
		return aug(x, self.preprocess), y

	def __len__(self):
		return len(self.dataset)


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]