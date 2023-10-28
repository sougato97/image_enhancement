import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def write_images(image, filename, dest): 

  # Set the new dimensions
  width = 600
  height = 400
  dim = (width, height)

  # Resize the image
  resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

  cv2.imwrite(dest + filename, resized)


def read_imgs(src, dest):
  '''Read all the images from the given folder'''
  for filename in os.listdir(src): # loop through the files in the folder
      if filename.endswith('.png') or filename.endswith('.jpg'): # check if the file is an image
          path = os.path.join(src, filename) # get the full path of the image
          image = cv2.imread(path) # read the image using cv2
          # Now you have the 'filename' and the corresponding 'image'
          write_images(image, filename, dest)


def main():
  src = '/home/sougato97/Thesis/datasets/earthcam/high/' # specify the folder name
  dest = '/home/sougato97/Thesis/datasets/earthcam/high/'
  # ratio = find_ratio(src)
  read_imgs(src, dest)

if __name__ == '__main__':
    main()

