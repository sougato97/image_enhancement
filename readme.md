#  Image Enhancement for Unconstrained Environments 
We worked on image denoising and exposure correction. Our method is to use a suitable generative neural network, train it on a custom version of a public dataset, and then evaluate it on different datasets. Our results show that training the model on our custom data helped it to maintain exposure as well as preserve features in a better way. Our paper has been accepted and is to be published in the IEEE Xplore. I will share the link once its finalized.

## Video demostration
<a href="https://youtu.be/SIXKh6Qd0nI">
   <img src="https://github.com/sougato97/image_enhancement/blob/master/readme_files/ppt_thumbnail.png" width="410" height="270" />
</a>
<br>
I have presented my paper at the 2023 Western New York Image and Signal Processing Workshop (WNYISPW). 
<br>

## Results on the Pepper data
<p float="left">
  <img src="https://github.com/sougato97/image_enhancement/blob/master/readme_files/custom_v3_img1.png" width="300" />
  <img src="https://github.com/sougato97/image_enhancement/blob/master/readme_files/custom_v3_img2.png" width="300" /> 
</p>
<br>
<p float="left">
  <img src="https://github.com/sougato97/image_enhancement/blob/master/readme_files/1_smallNet.jpg" width="300" />
  <img src="https://github.com/sougato97/image_enhancement/blob/master/readme_files/2_smallNet.jpg" width="300" /> 
</p>
<br>
<p float="left">
  <img src="https://github.com/sougato97/image_enhancement/blob/master/datasets/pepper/low/1.png" width="300" />
  <img src="https://github.com/sougato97/image_enhancement/blob/master/datasets/pepper/low/2.png" width="300" /> 
</p>
<br>
Images in the bottom row were fed into different models for image enhancement. The top two images are from our custom model, and the images in 2nd row are from the LLFLow author's smallNet model. We found the smallNet variant to be performing better than the other pretrained ones. Its evident that the custom model handles exposure better and also preserves more features. Our results are verified in the experimental results section in the ppt/video demo/paper(to be shared). 

## Referred Codebases
[LLFLOW](https://github.com/wyf0912/LLFlow) <br>
[Unprocessing](https://github.com/timothybrooks/unprocessing) <br>
[Unprocessing_Pytorch](https://github.com/aasharma90/UnprocessDenoising_PyTorch)

## How to execute the code 
- Go to the folder LLFlow/code/confs, modify the .yml files as required. Better check the documentations from LLFLOW readme file. 

## Datasets used
- [VE-LOL](https://flyywh.github.io/IJCV2021LowLight_VELOL/)
- Custom datasets based on [LOL](https://daooshee.github.io/BMVC2018website/)
- [EarthCam](https://www.earthcam.com/usa/newyork/worldtradecenter/?cam=skyline_g)
- [pepper & pepper_v2](https://github.com/sougato97/image_enhancement/tree/master/datasets): captured using SoftBank Robotics [Pepper Robot](https://us.softbankrobotics.com/pepper)

## My setup 
- I am using windows 11 (WSL - Ubuntu 18)
- For gpu setup please install, nvidia cuda toolkit on windows 
- Your distro will be able to access the gpu drivers

## Installation
To install this project, follow these steps:
- Install Miniconda (https://docs.conda.io/en/latest/miniconda.html)
- Create the conda environment using my requirements.txt. There are a lot of libraries which might me irrelevant for your case, for that you may simply follow the [pytorch installation](https://pytorch.org/). 

