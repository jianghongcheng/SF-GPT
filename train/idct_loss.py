import numpy as np
import matplotlib.image as mpimg
from numpy import empty,arange,exp,real,imag,pi
from numpy.fft import rfft,irfft
import matplotlib.pyplot as plt
import torch
from dct_loss import dct_loss
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

# torch.manual_seed(2)   #为CPU设置种子用于生成随机数，以使得结果是确定的
# img = torch.rand(1, 3, 256,256)
# DC, AC_H, AC_V, DG = dct_loss(img)

def idct_loss(img, DC, AC_H, AC_V, DG):
    patch = 4
    dct_2d = ImageDCT(patch)
    img = img.squeeze().cpu().numpy()
    img = img.transpose((1,2,0))
    dct = np.zeros_like(img, dtype=np.float32)


    DC = DC.squeeze(dim=0).cpu().detach().numpy().tolist()
    AC_H = AC_H.squeeze(dim=0).cpu().detach().numpy().tolist()
    AC_V = AC_V.squeeze(dim=0).cpu().detach().numpy().tolist()
    DG = DG.squeeze(dim=0).cpu().detach().numpy().tolist()

    [dct[::patch, ::patch, 0], dct[::patch, ::patch, 1], dct[::patch, ::patch, 2]] = DC
    [dct[1::patch, ::patch, 0], dct[1::patch, ::patch, 1], dct[1::patch, ::patch, 2], dct[2::patch, ::patch, 0],
        dct[2::patch, ::patch, 1], dct[2::patch, ::patch, 2], dct[3::patch, ::patch, 0], dct[3::patch, ::patch, 1],
        dct[3::patch, ::patch, 2]] = AC_H
    [dct[::patch, 1::patch, 0], dct[::patch, 1::patch, 1], dct[::patch, 1::patch, 2], dct[::patch, 2::patch, 0],
        dct[::patch, 2::patch, 1], dct[::patch, 2::patch, 2], dct[::patch, 3::patch, 0], dct[::patch, 3::patch, 1],
        dct[::patch, 3::patch, 2]] = AC_V
    [dct[1::patch, 1::patch, 0], dct[1::patch, 1::patch, 1], dct[1::patch, 1::patch, 2],
        dct[1::patch, 2::patch, 0], dct[1::patch, 2::patch, 1], dct[1::patch, 2::patch, 2],
        dct[1::patch, 3::patch, 0], dct[1::patch, 3::patch, 1], dct[1::patch, 3::patch, 2],
        dct[2::patch, 1::patch, 0], dct[2::patch, 1::patch, 1], dct[2::patch, 1::patch, 2],
        dct[2::patch, 2::patch, 0], dct[2::patch, 2::patch, 1], dct[2::patch, 2::patch, 2],
        dct[2::patch, 3::patch, 0], dct[2::patch, 3::patch, 1], dct[2::patch, 3::patch, 2],
        dct[3::patch, 1::patch, 0], dct[3::patch, 1::patch, 1], dct[3::patch, 1::patch, 2],
        dct[3::patch, 2::patch, 0], dct[3::patch, 2::patch, 1], dct[3::patch, 2::patch, 2],
        dct[3::patch, 3::patch, 0], dct[3::patch, 3::patch, 1], dct[3::patch, 3::patch, 2]] = DG

    idct_img = dct_2d.idct_2d(dct)
    return idct_img
