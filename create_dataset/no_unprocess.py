#!/sougatob/miniconda3/envs/nemo/bin/python3
import torch
import cv2
import torchvision.transforms as transforms
from unprocess import *
from process import *
import matplotlib.pyplot as plt
# from PIL import Image
import numpy as np
import os

'''We have to convert the low images to RAW'''


def change_brightness(image):
  # Adjust the brightness by a factor (positive or negative)
  brightness_factor = 0.2  # Decrease Brightness by 50%
  adjusted_image = image * brightness_factor

  # Clamp the pixel values to the valid range of 0-1
  adjusted_image = np.clip(adjusted_image, 0, 1)

  return adjusted_image

def write_images(image, filename, dest):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB color space    

  # Define the transform
  transform = transforms.ToTensor()  

  # Convert the image to tensor
  tensor = transform(image)

  # Add noise to the image
  shot_noise, read_noise = random_noise_levels()
  noisy_img = add_noise(tensor, shot_noise, read_noise)

  # Convert the tensor image to a numpy array
  numpy_image = noisy_img.numpy()

  # Permute the dimensions of the numpy array
  numpy_image = numpy_image.transpose(1, 2, 0)
  
  # Change the brightness by a particular percentage
  numpy_image = change_brightness(numpy_image)

  # Write the image into the destination folder
  plt.imsave(dest + filename, numpy_image)


def find_ratio(src):
  sum = 0
  for filename in os.listdir(src): # loop through the files in the folder
      if filename.endswith('.png') or filename.endswith('.jpg'): # check if the file is an image
          path = os.path.join(src, filename) # get the full path of the image
          high_img = cv2.imread(path) # read the image using cv2
          low_path = src[:-5] + 'low/' + filename
          low_img = cv2.imread(low_path)



def read_imgs(src, dest):
  '''Read all the properly exposed images from the given folder'''
  for filename in os.listdir(src): # loop through the files in the folder
      if filename.endswith('.png') or filename.endswith('.jpg'): # check if the file is an image
          path = os.path.join(src, filename) # get the full path of the image
          image = cv2.imread(path) # read the image using cv2
          # Now you have the 'filename' and the corresponding 'image'
          write_images(image, filename, dest)


def main():
  src = '/home/sougato97/Thesis/datasets/LOLdataset/our485/high/' # specify the folder name
  dest = '/home/sougato97/Thesis/datasets/LOLdataset/no_unprocess_Noisy/low/'
  # ratio = find_ratio(src)
  read_imgs(src, dest)


if __name__ == '__main__':
    main()

