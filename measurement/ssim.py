from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import cv2
from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)
I1 = cv2.imread(
    '/media/max/a/DWT2021/Semantic-Colorization-GAN-main_HC (LL_sub)/PSNR/output/test_npu/Testing_0001_nir_reg.png')
I2 = cv2.imread(
    '/media/max/a/DWT2021/Semantic-Colorization-GAN-main_HC (LL_sub)/PSNR/output/test/Testing_0001_nir_reg.png')
# I2 = cv2.imread('./blur.png')
# I2 = cv2.resize(I2, I1.shape[0:2])
# print(I1.shape, I2.shape) # returns (256,256,3)

# tensors
X = torch.from_numpy(np.rollaxis(I1, 2)).float().unsqueeze(0)
Y = torch.from_numpy(np.rollaxis(I2, 2)).float().unsqueeze(0)
# calculate ssim & ms-ssim for each image
ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# reuse the gaussian kernel with SSIM & MS_SSIM.
ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)
print(ms_ssim_val)