#! /home/csgrad/sougatob/miniconda3/envs/nemo/bin/python3
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

def write_unprocess_images(image, filename, dest):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB color space    
  # Define the transform
  transform = transforms.ToTensor()  
  # Convert the image to tensor
  tensor = transform(image)
  un_image, metadata = unprocess(tensor)
  # Convert the tensor image to a numpy array
  numpy_image = un_image.numpy()

  # Permute the dimensions of the numpy array
  numpy_image = numpy_image.transpose(1, 2, 0)
  # Write the image into the destination folder
  # print("The name of the destination is: ", dest + filename)
  # print('The shape of the array is: ', numpy_image.shape)
  plt.imsave(dest + filename, numpy_image)

def read_imgs(src, dest):
  '''Read all the properly exposed images from the given folder'''
  for filename in os.listdir(src): # loop through the files in the folder
      if filename.endswith('.png') or filename.endswith('.jpg'): # check if the file is an image
          path = os.path.join(src, filename) # get the full path of the image
          image = cv2.imread(path) # read the image using cv2
          # Now you have the 'filename' and the corresponding 'image'
          write_unprocess_images(image, filename, dest)


def main():
  src = '/home/csgrad/sougatob/ImUncon/datasets/LOLdataset/our485/low/' # specify the folder name
  dest = '/home/csgrad/sougatob/ImUncon/datasets/LOLdataset/RAW_NoNoiseHigh/low/'
  read_imgs(src, dest)


if __name__ == '__main__':
    main()

