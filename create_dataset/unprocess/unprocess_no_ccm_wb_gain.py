# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unprocesses sRGB images into realistic raw data.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
import torch.distributions as tdist
import cv2
import torchvision.transforms as transforms



def inverse_smoothstep(image):
  """Approximately inverts a global tone mapping curve."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  image = torch.clamp(image, min=0.0, max=1.0)
  out   = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0) 
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def gamma_expansion(image):
  """Converts from gamma to linear space."""
  # Clamps to prevent numerical instability of gradients near zero.
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  out   = torch.clamp(image, min=1e-8) ** 2.2
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out



#slicing notation [start:stop:step]
# 0::2 means select the 2nd element starting from the 0th address
# 1::2 means select the 2nd element starting from the 1st address
def mosaic(image):
  """Extracts RGGB Bayer planes from an RGB image."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  shape = image.size()
  red   = image[0::2, 0::2, 0] 
  green_red  = image[0::2, 1::2, 1]
  green_blue = image[1::2, 0::2, 1]
  blue = image[1::2, 1::2, 2]
  out  = torch.stack((red, green_red, green_blue, blue), dim=-1)
  out  = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4)) # will work even if we dont use this
  out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out



def unprocess(image):
  """Unprocesses an image from sRGB to realistic raw data."""

  # Approximately inverts global tone mapping.
  image = inverse_smoothstep(image)
  # Inverts gamma compression.
  image = gamma_expansion(image)
  # Clips saturated pixels.
  image = torch.clamp(image, min=0.0, max=1.0)
  # Applies a Bayer mosaic.
  image = mosaic(image)
  # image = mosaic_using_opencv(image)
  return image


def random_noise_levels():
  """Generates random noise levels from a log-log linear distribution."""
  log_min_shot_noise = np.log(0.0001)
  log_max_shot_noise = np.log(0.012)
  log_shot_noise     = torch.FloatTensor(1).uniform_(log_min_shot_noise, log_max_shot_noise)
  shot_noise = torch.exp(log_shot_noise)

  line = lambda x: 2.18 * x + 1.20
  n    = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26])) 
  log_read_noise = line(log_shot_noise) + n.sample()
  read_noise     = torch.exp(log_read_noise)
  return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
  """Adds random shot (proportional to image) and read (independent) noise."""
  image    = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  variance = image * shot_noise + read_noise
  n        = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance)) 
  noise    = n.sample()
  out      = image + noise
  out      = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


