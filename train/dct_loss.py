import numpy as np
import matplotlib.image as mpimg
from numpy import empty,arange,exp,real,imag,pi
from numpy.fft import rfft,irfft
import matplotlib.pyplot as plt
import torch

import cv2
# Import functions and libraries
import numpy as np
from numpy import r_
import matplotlib.pyplot as plt
from scipy import fftpack
class ImageDCT():
    def __init__(self, dct_block=8):
        self.dct_block = dct_block
    def dct2(self, a):
        return fftpack.dct(fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')
    def idct2(self, a):
        return fftpack.idct(fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')
    def dct_2d(self, img):
        imsize = img.shape
        dct = np.zeros(imsize)
        # Do DCT on cjhannel 1
        for i in r_[:imsize[0]:self.dct_block]:
            for j in r_[:imsize[1]:self.dct_block]:
                if img.ndim == 2:
                    dct[i:(i+self.dct_block), j:(j+self.dct_block)] = self.dct2(img[i:(i+self.dct_block), j:(j+self.dct_block)])
                if img.ndim == 3:
                    for k in range(3):
                        dct[i:(i+self.dct_block), j:(j+self.dct_block), k] = self.dct2(img[i:(i+self.dct_block), j:(j+self.dct_block), k])
        return dct
    def idct_2d(self, img):
        imsize = img.shape
        idct = np.zeros(imsize)
        # Do DCT on cjhannel 1
        for i in r_[:imsize[0]:self.dct_block]:
            for j in r_[:imsize[1]:self.dct_block]:
                if img.ndim == 2:
                    idct[i:(i+self.dct_block), j:(j+self.dct_block)] = self.idct2(img[i:(i+self.dct_block), j:(j+self.dct_block)])
                if img.ndim == 3:
                    for k in range(3):
                        idct[i:(i+self.dct_block), j:(j+self.dct_block), k] = self.idct2(img[i:(i+self.dct_block), j:(j+self.dct_block), k])
        return idct




def dct_loss(img):
    patch = 4
    img=img.cpu()
    BATCH_DC = []
    BATCH_H = []
    BATCH_V = []
    BATCH_DG = []

    for t in range(img.shape[0]):
        im_i = img[t,:,:,:]
        im = im_i.transpose(0,1).transpose(1,2).detach().numpy()

        dct_2d = ImageDCT(patch)
        dct = dct_2d.dct_2d(im)
        DC =[dct[::patch,::patch,0], dct[::patch,::patch,1], dct[::patch,::patch,2]]
        H = [dct[1::patch,::patch,0], dct[1::patch,::patch,1], dct[1::patch,::patch,2], dct[2::patch,::patch,0], dct[2::patch,::patch,1],dct[2::patch,::patch,2],dct[3::patch,::patch,0], dct[3::patch,::patch,1], dct[3::patch,::patch,2]]
        V = [dct[::patch,1::patch,0], dct[::patch,1::patch,1], dct[::patch,1::patch,2], dct[::patch,2::patch,0], dct[::patch,2::patch,1],dct[::patch,2::patch,2],dct[::patch,3::patch,0], dct[::patch,3::patch,1], dct[::patch,3::patch,2]]
        DG = [dct[1::patch,1::patch,0], dct[1::patch,1::patch,1], dct[1::patch,1::patch,2], dct[1::patch,2::patch,0], dct[1::patch,2::patch,1], dct[1::patch,2::patch,2], dct[1::patch,3::patch,0],dct[1::patch,3::patch,1],dct[1::patch,3::patch,2],
              dct[2::patch,1::patch,0], dct[2::patch,1::patch,1], dct[2::patch,1::patch,2], dct[2::patch,2::patch,0], dct[2::patch,2::patch,1], dct[2::patch,2::patch,2], dct[2::patch,3::patch,0],dct[2::patch,3::patch,1],dct[2::patch,3::patch,2],
              dct[3::patch,1::patch,0], dct[3::patch,1::patch,1], dct[3::patch,1::patch,2], dct[3::patch,2::patch,0], dct[3::patch,2::patch,1], dct[3::patch,2::patch,2], dct[3::patch,3::patch,0], dct[3::patch,3::patch,1], dct[3::patch,3::patch,2]]
        BATCH_DC.append(DC)
        BATCH_H.append(H)
        BATCH_V.append(V)
        BATCH_DG.append(DG)

    DC_array = np.array(BATCH_DC)
    BATCH_DC_tensor = torch.from_numpy(DC_array).cuda()
    H_array = np.array(BATCH_H)
    BATCH_H_tensor = torch.from_numpy(H_array).cuda()
    V_array = np.array(BATCH_V)
    BATCH_V_tensor = torch.from_numpy(V_array).cuda()
    DG_array = np.array(BATCH_DG)
    BATCH_DG_tensor = torch.from_numpy(DG_array).cuda()

    return BATCH_DC_tensor, BATCH_H_tensor,BATCH_V_tensor,BATCH_DG_tensor

# torch.manual_seed(2)   #为CPU设置种子用于生成随机数，以使得结果是确定的
# img = torch.rand(8, 1, 256,256)
# img = torch.cat((img,img,img),1)
# true_DC, true_H, true_V, true_DG = dct_loss(img)
#
# nir_d = true_DC[:,0:1,:,:]
# nir_h = torch.cat((true_H[:,0:1,:,:], true_H[:,3:4,:,:], true_H[:,6:7,:,:]),1)
# nir_v = torch.cat((true_V[:,0:1,:,:], true_V[:,3:4,:,:], true_V[:,6:7,:,:]),1)
# nir_g = torch.cat((true_DG[:,0:1,:,:], true_DG[:,3:4,:,:], true_DG[:,6:7,:,:], true_DG[:,9:10,:,:],true_DG[:,12:13,:,:],true_DG[:,15:16,:,:],true_DG[:,18:19,:,:],true_DG[:,21:22,:,:],true_DG[:,24:25,:,:]),1)
# nir_all = torch.cat((nir_d,nir_h,nir_v,nir_g),1)
# print(nir_all.shape)
#
