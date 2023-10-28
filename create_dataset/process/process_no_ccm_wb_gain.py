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

"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import cv2
import torchvision.transforms as transforms

# bayer_images = bayer_images.permute(0, 2, 3, 1) # Permute the image tensor to BxHxWxC format from BxCxHxW format
# BxHxWxC is a notation that describes the shape of a 4D tensor that represents a batch of images in PyTorch. It stands for:
# B: Batch size, the number of images in the batch.
# H: Height, the number of pixels along the vertical axis of each image.
# W: Width, the number of pixels along the horizontal axis of each image.
# C: Channels, the number of color channels in each image. For example, C=3 for RGB images and C=1 for grayscale images.


def demosaic(bayer_images):
  def SpaceToDepth_fact2(x):
    # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
    bs = 2
    N, C, H, W = x.size()
    x = x.view(N, C, H // bs, bs, W // bs, bs)      # (N, C, H//bs, bs, W//bs, bs)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()    # (N, bs, bs, C, H//bs, W//bs)
    x = x.view(N, C * (bs ** 2), H // bs, W // bs)  # (N, C*bs^2, H//bs, W//bs)
    return x
  def DepthToSpace_fact2(x):
    # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
    bs = 2
    N, C, H, W = x.size()
    x = x.view(N, bs, bs, C // (bs ** 2), H, W)     # (N, bs, bs, C//bs^2, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()    # (N, C//bs^2, H, bs, W, bs)
    x = x.view(N, C // (bs ** 2), H * bs, W * bs)   # (N, C//bs^2, H * bs, W * bs)
    return x

  """Bilinearly demosaics a batch of RGGB Bayer images."""
  bayer_images = bayer_images.permute(0, 2, 3, 1) # Permute the image tensor to BxHxWxC format from BxCxHxW format

  shape = bayer_images.size()
  shape = [shape[1] * 2, shape[2] * 2]

  red = bayer_images[Ellipsis, 0:1]
  upsamplebyX = nn.Upsample(size=shape, mode='bilinear', align_corners=False)
  red = upsamplebyX(red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

  green_red = bayer_images[Ellipsis, 1:2]
  green_red = torch.flip(green_red, dims=[1]) # Flip left-right
  green_red = upsamplebyX(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
  green_red = torch.flip(green_red, dims=[1]) # Flip left-right
  green_red = SpaceToDepth_fact2(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

  green_blue = bayer_images[Ellipsis, 2:3]
  green_blue = torch.flip(green_blue, dims=[0]) # Flip up-down
  green_blue = upsamplebyX(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
  green_blue = torch.flip(green_blue, dims=[0]) # Flip up-down
  green_blue = SpaceToDepth_fact2(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

  green_at_red = (green_red[Ellipsis, 0] + green_blue[Ellipsis, 0]) / 2
  green_at_green_red = green_red[Ellipsis, 1]
  green_at_green_blue = green_blue[Ellipsis, 2]
  green_at_blue = (green_red[Ellipsis, 3] + green_blue[Ellipsis, 3]) / 2

  green_planes = [
      green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
  ]
  green = DepthToSpace_fact2(torch.stack(green_planes, dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

  blue = bayer_images[Ellipsis, 3:4]
  blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])
  blue = upsamplebyX(blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
  blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])

  rgb_images = torch.cat([red, green, blue], dim=-1)
  rgb_images = rgb_images.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
  return rgb_images


def gamma_compression(images, gamma=2.2):
  """Converts from linear to gamma space."""
  # Clamps to prevent numerical instability of gradients near zero.
  images = images.permute(1, 2, 0) # Permute the image tensor to BxHxWxC format from BxCxHxW format
  outs   = torch.clamp(images, min=1e-8) ** (1.0 / gamma)
  outs   = outs.permute(2, 0, 1)  # Re-Permute the tensor back to BxCxHxW format
  return outs

def process(bayer_image):
  """Processes a batch of Bayer RGGB image into sRGB image."""
  # Demosaic.
  bayer_image = torch.clamp(bayer_image, min=0.0, max=1.0)
  # For this demosaic to work, it needs to be a 4D matrix, so expanding
  shape = bayer_image.size()
  bayer_image = bayer_image.view(1, shape[0], shape[1], shape[2])
  image = demosaic(bayer_image)
  image = image[0] # Again converting to 3D matrix
  # Gamma compression.
  image = torch.clamp(image, min=0.0, max=1.0)
  image = gamma_compression(image)
  return image
