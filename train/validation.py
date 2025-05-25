import argparse
import os
import torch
from torch.utils.data import DataLoader
from numpy import *

import utils
import dataset
from dct_loss import dct_loss
from idct_loss import idct_loss


import matplotlib.pyplot as plt

import numpy as np

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
# from community import fid

eps = 1e-6

def _binarize(y_data, threshold):
    """
    args:
        y_data : [float] 4-d tensor in [batch_size, channels, img_rows, img_cols]
        threshold : [float] [0.0, 1.0]
    return 4-d binarized y_data
    """
    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data
class PSNR(object):
    def __init__(self, des="Peak Signal to Noise Ratio"):
        self.des = des

    def __repr__(self):
        return "PSNR"

    def __call__(self, y_pred, y_true, dim=1, threshold=None):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        if threshold:
            y_pred = _binarize(y_pred, threshold)
        mse = torch.mean((y_pred - y_true) ** 2)
        return 10 * torch.log10(1 / mse)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument('--val_path', type = str, default = './validation_results', help = 'save the validation results to certain path')
    parser.add_argument('--load_name', type = str, default = './models/4_noGAN_epoch500_bs16.pth', help = 'load the pre-trained model with certain epoch')#####
    # ./models/11/SCGAN_noGAN_epoch1000_bs4.pth

    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # network parameters

    parser.add_argument('--in_channels', type = int, default = 1, help = 'in channel for U-Net encoder')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channel for U-Net decoder')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channel for U-Net decoder')
    parser.add_argument('--latent_channels', type = int, default = 128, help = 'start channel for APN')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation function for generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation function for discriminator')
    parser.add_argument('--norm_g', type = str, default = 'bn', help = 'normalization type for generator')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type for discriminator')
    # VCIP dataset
    parser.add_argument('--baseroot_rgb', type = str, default = './dataset/VCIP_20_aug/test_rgb', help = 'color image baseroot')
    parser.add_argument('--baseroot_gray', type = str, default = './dataset/VCIP_20_aug/test_nir', help = 'saliency map baseroot')
    parser.add_argument('--baseroot_rgb_test', type = str, default = './dataset/VCIP_20_aug/test_rgb', help = 'color image baseroot')#keep same name
    parser.add_argument('--baseroot_gray_test', type = str, default = './dataset/VCIP_20_aug/test_nir', help = 'nir image baseroot')####ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ»ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ nir
    # # ARRI dataset
    # parser.add_argument('--baseroot_rgb', type = str, default = './dataset/ARRI_256/RGB_Mountain_test', help = 'color image baseroot')#keep same name
    # parser.add_argument('--baseroot_gray', type = str, default = './dataset/ARRI_256/NIR_Mountain_test', help = 'nir image baseroot')####ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ»ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ nir
    # parser.add_argument('--baseroot_rgb_test', type = str, default = './dataset/ARRI_256/RGB_Mountain_test', help = 'color image baseroot')#keep same name
    # parser.add_argument('--baseroot_gray_test', type = str, default = './dataset/ARRI_256/NIR_Mountain_test', help = 'nir image baseroot')####ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ»ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ nir
    


    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    # RCAN
    parser.add_argument('--num_rg', type=int, default=4)
    parser.add_argument('--network_mode', type = str, default = 'mst', help = 'type of network: [mst | hat ], mst is recommended')

    opt = parser.parse_args()
    print(opt)
    utils.check_path(opt.val_path)
    
    # Define the network
    generator = utils.create_generator(opt)
    generator = generator.cuda()

    # Define the dataset
    trainset = dataset.ColorizationDataset_Val(opt)
    print('The overall number of images:', len(trainset))
    DWT_PSNR = []
    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    metric_PSNR = PSNR()

    # For loop training
    for i, (true_L, true_RGB,file_name) in enumerate(dataloader):
        true_L = true_L.cuda()
        true_RGB = true_RGB.cuda()
        with torch.no_grad():
            Whole_fake = generator(true_L)
            print("Total number of model parameter: %d", count_parameters(generator))
            # print(Whole_fake.shape)
            dwt_img = Whole_fake
        ############################################################################################
        # print(dwt_img.shape)
        dwt_img = np.squeeze(dwt_img).cpu().detach().numpy()
        # print(dwt_img.shape)
        dwt_img = dwt_img.transpose(1,2,0)
        # print(dwt_img.shape)
        import cv2
        # cv2.imshow('window_title', dct_img)
        # dwt_img = dwt_img.astype(np.float32)
        dwt_img_255 = cv2.cvtColor(dwt_img, cv2.COLOR_BGR2RGB);
        dwt_img_255 = dwt_img_255 *255
        
        print(file_name[0])


        cv2.imwrite(f"../PSNR/output/test/{file_name[0]}", dwt_img_255)

        ##############################################################################################################################################
    #     print("dwt psnr",i+1)
    #     PSNR_VCIP = metric_PSNR(dwt_img, true_RGB).item()
    #     print(PSNR_VCIP)
    #     DWT_PSNR.append(PSNR_VCIP)

    # print("mean_dwt_psnr")
    # print(mean(DWT_PSNR))
