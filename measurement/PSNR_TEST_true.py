import os
import numpy as np
import math
import cv2
import argparse
import os
import torch
from torch.utils.data import DataLoader
from numpy import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from community import lpips
# from community import fid

# # import utils
# import dataset
# from dct_loss import dct_loss
# from idct_loss import idct_loss
# from dct_loss import dct_loss


import matplotlib.pyplot as plt

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


def _argmax(y_data, dim):
    """
    args:
        y_data : 4-d tensor in [batch_size, chs, img_rows, img_cols]
        dim : int
    return 3-d [int] y_data
    """
    return torch.argmax(y_data, dim).int()


def _get_tp(y_pred, y_true):
    """
    args:
        y_true : [int] 3-d in [batch_size, img_rows, img_cols]
        y_pred : [int] 3-d in [batch_size, img_rows, img_cols]
    return [float] true_positive
    """
    return torch.sum(y_true * y_pred).float()


def _get_fp(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_positive
    """
    return torch.sum((1 - y_true) * y_pred).float()


def _get_tn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] true_negative
    """
    return torch.sum((1 - y_true) * (1 - y_pred)).float()


def _get_fn(y_pred, y_true):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        y_pred : 3-d ndarray in [batch_size, img_rows, img_cols]
    return [float] false_negative
    """
    return torch.sum(y_true * (1 - y_pred)).float()


def _get_weights(y_true, nb_ch):
    """
    args:
        y_true : 3-d ndarray in [batch_size, img_rows, img_cols]
        nb_ch : int
    return [float] weights
    """
    batch_size, img_rows, img_cols = y_true.shape
    pixels = batch_size * img_rows * img_cols
    weights = [torch.sum(y_true == ch).item() / pixels for ch in range(nb_ch)]
    return weights


class CFMatrix(object):
    def __init__(self, des=None):
        self.des = des

    def __repr__(self):
        return "ConfusionMatrix"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return confusion matrix
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_tn = _get_tn(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            mperforms = [nb_tp, nb_fp, nb_tn, nb_fn]
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 4).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_tn = _get_tn(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                performs[int(ch), :] = [nb_tp, nb_fp, nb_tn, nb_fn]
            mperforms = sum([i * j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class OAAcc(object):
    def __init__(self, des="Overall Accuracy"):
        self.des = des

    def __repr__(self):
        return "OAcc"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return (tp+tn)/total
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)

        nb_tp_tn = torch.sum(y_true == y_pred).float()
        mperforms = nb_tp_tn / (batch_size * img_rows * img_cols)
        performs = None
        return mperforms, performs


class Precision(object):
    def __init__(self, des="Precision"):
        self.des = des

    def __repr__(self):
        return "Prec"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return tp/(tp+fp)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            mperforms = nb_tp / (nb_tp + nb_fp + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                performs[int(ch)] = nb_tp / (nb_tp + nb_fp + esp)
            mperforms = sum([i * j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class Recall(object):
    def __init__(self, des="Recall"):
        self.des = des

    def __repr__(self):
        return "Reca"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return tp/(tp+fn)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            mperforms = nb_tp / (nb_tp + nb_fn + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                performs[int(ch)] = nb_tp / (nb_tp + nb_fn + esp)
            mperforms = sum([i * j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class F1Score(object):
    def __init__(self, des="F1Score"):
        self.des = des

    def __repr__(self):
        return "F1Sc"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return 2*precision*recall/(precision+recall)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            _precision = nb_tp / (nb_tp + nb_fp + esp)
            _recall = nb_tp / (nb_tp + nb_fn + esp)
            mperforms = 2 * _precision * _recall / (_precision + _recall + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                _precision = nb_tp / (nb_tp + nb_fp + esp)
                _recall = nb_tp / (nb_tp + nb_fn + esp)
                performs[int(ch)] = 2 * _precision * \
                                    _recall / (_precision + _recall + esp)
            mperforms = sum([i * j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class Kappa(object):
    def __init__(self, des="Kappa"):
        self.des = des

    def __repr__(self):
        return "Kapp"

    def __call__(self, y_pred, y_true, threshold=0.5):

        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return (Po-Pe)/(1-Pe)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            nb_tp = _get_tp(y_pred, y_true)
            nb_fp = _get_fp(y_pred, y_true)
            nb_tn = _get_tn(y_pred, y_true)
            nb_fn = _get_fn(y_pred, y_true)
            nb_total = nb_tp + nb_fp + nb_tn + nb_fn
            Po = (nb_tp + nb_tn) / nb_total
            Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn) +
                  (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total ** 2)
            mperforms = (Po - Pe) / (1 - Pe + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                nb_tp = _get_tp(y_pred_ch, y_true_ch)
                nb_fp = _get_fp(y_pred_ch, y_true_ch)
                nb_tn = _get_tn(y_pred_ch, y_true_ch)
                nb_fn = _get_fn(y_pred_ch, y_true_ch)
                nb_total = nb_tp + nb_fp + nb_tn + nb_fn
                Po = (nb_tp + nb_tn) / nb_total
                Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn)
                      + (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total ** 2)
                performs[int(ch)] = (Po - Pe) / (1 - Pe + esp)
            mperforms = sum([i * j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class Jaccard(object):
    def __init__(self, des="Jaccard"):
        self.des = des

    def __repr__(self):
        return "Jacc"

    def __call__(self, y_pred, y_true, threshold=0.5):
        """
        args:
            y_true : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, chs, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return intersection / (sum-intersection)
        """
        batch_size, chs, img_rows, img_cols = y_true.shape
        device = y_true.device
        if chs == 1:
            y_pred = _binarize(y_pred, threshold)
            y_true = _binarize(y_true, threshold)
            _intersec = torch.sum(y_true * y_pred).float()
            _sum = torch.sum(y_true + y_pred).float()
            mperforms = _intersec / (_sum - _intersec + esp)
            performs = None
        else:
            y_pred = _argmax(y_pred, 1)
            y_true = _argmax(y_true, 1)
            performs = torch.zeros(chs, 1).to(device)
            weights = _get_weights(y_true, chs)
            for ch in range(chs):
                y_true_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_pred_ch = torch.zeros(batch_size, img_rows, img_cols)
                y_true_ch[y_true == ch] = 1
                y_pred_ch[y_pred == ch] = 1
                _intersec = torch.sum(y_true_ch * y_pred_ch).float()
                _sum = torch.sum(y_true_ch + y_pred_ch).float()
                performs[int(ch)] = _intersec / (_sum - _intersec + esp)
            mperforms = sum([i * j for (i, j) in zip(performs, weights)])
        return mperforms, performs


class MSE(object):
    def __init__(self, des="Mean Square Error"):
        self.des = des

    def __repr__(self):
        return "MSE"

    def __call__(self, y_pred, y_true, dim=1, threshold=None):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return mean_squared_error, smaller the better
        """
        if threshold:
            y_pred = _binarize(y_pred, threshold)
        return torch.mean((y_pred - y_true) ** 2)


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


class SSIM(object):
    '''
    modified from https://github.com/jorge-pessoa/pytorch-msssim
    '''

    def __init__(self, des="structural similarity index"):
        self.des = des

    def __repr__(self):
        return "SSIM"

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(w_size)])
        return gauss / gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret


class LPIPS(object):
    '''
    borrowed from https://github.com/richzhang/PerceptualSimilarity
    '''

    def __init__(self, cuda, des="Learned Perceptual Image Patch Similarity", version="0.1"):
        self.des = des
        self.version = version
        self.model = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=cuda)

    def __repr__(self):
        return "LPIPS"

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        return self.model.forward(y_pred, y_true)


class AE(object):
    """
    Modified from matlab : colorangle.m, MATLAB V2019b
    angle = acos(RGB1' * RGB2 / (norm(RGB1) * norm(RGB2)));
    angle = 180 / pi * angle;
    """

    def __init__(self, des='average Angular Error'):
        self.des = des

    def __repr__(self):
        return "AE"

    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        return average AE, smaller the better
        """
        dotP = torch.sum(y_pred * y_true, dim=1)
        Norm_pred = torch.sqrt(torch.sum(y_pred * y_pred, dim=1))
        Norm_true = torch.sqrt(torch.sum(y_true * y_true, dim=1))
        ae = 180 / math.pi * torch.acos(dotP / (Norm_pred * Norm_true + eps))
        return ae.mean(1).mean(1)





from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import cv2


def ssim(image1, image2, K, window_size, L):
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5  # default
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    # define constants
    # * L = 255 for constants doesn't produce meaningful results; thus L = 1
    # C1 = (K[0]*L)**2;
    # C2 = (K[1]*L)**2;
    C1 = K[0] ** 2;
    C2 = K[1] ** 2;

    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


psnr = PSNR()
# def psnr(img1, img2):
#     mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
#     if mse < 1.0e-10:
#         return 100 * 1.0
#     return 10 * math.log10(255.0 * 255.0 / mse)

ae = AE()





root1 = path1 = './output/gt/'  # Ã§Â½ÂÃ§Â»ÂÃ¨Â¾ÂÃ¥ÂÂºÃ¥ÂÂÃ§ÂÂÃ¥ÂÂ¾Ã¥ÂÂ##11 is best
root2 = path2 = './output/test/'  # gt
# root3 = path3 = 'output/db1/' #GT
f_nums = len(os.listdir(path1))
print(f_nums)
list_psnr = []
list_ssim = []
list_ae = []

#Ã¦ÂÂÃ§ÂÂ§Ã¦ÂÂ°Ã¥Â­ÂÃ§ÂÂÃ¥Â¤Â§Ã¥Â°ÂÃ¯Â¼ÂÃ¨Â¿ÂÃ¨Â¡ÂÃ¦ÂÂÃ¥ÂºÂ

path1 = os.listdir(path1)
# path3.sort(key=lambda x: int(x.replace("validation_","").split('.')[0]))#Ã¦ÂÂÃ§ÂÂ§Ã¦ÂÂ°Ã¥Â­ÂÃ¨Â¿ÂÃ¨Â¡ÂÃ¦ÂÂÃ¥ÂºÂÃ¥ÂÂÃ¦ÂÂÃ©Â¡ÂºÃ¥ÂºÂÃ¨Â¯Â»Ã¥ÂÂÃ¦ÂÂÃ¤Â»Â¶Ã¥Â¤Â¹Ã¤Â¸ÂÃ§ÂÂÃ¥ÂÂ¾Ã§ÂÂ
# path1.sort(key=lambda x: int(x[8:-12]))
print(path1)

path2 = os.listdir(path2)
# path3.sort(key=lambda x: int(x.replace("validation_","").split('.')[0]))#Ã¦ÂÂÃ§ÂÂ§Ã¦ÂÂ°Ã¥Â­ÂÃ¨Â¿ÂÃ¨Â¡ÂÃ¦ÂÂÃ¥ÂºÂÃ¥ÂÂÃ¦ÂÂÃ©Â¡ÂºÃ¥ÂºÂÃ¨Â¯Â»Ã¥ÂÂÃ¦ÂÂÃ¤Â»Â¶Ã¥Â¤Â¹Ã¤Â¸ÂÃ§ÂÂÃ¥ÂÂ¾Ã§ÂÂ
# path2.sort(key=lambda x: int(x[8:-12]))
print(path2)



# img_a = cv2.imread(root1 + path1[0])
# img_b = cv2.imread(root3 + path3[0])
#
# print(img_b.shape)


for i in range(f_nums):#Ã¤Â»Â0Ã¥Â¼ÂÃ¥Â§Â
    img_a = cv2.imread(root1 + path1[i])/255
    img_b = cv2.imread(root2 + path2[i])/255
    psnr_num = psnr(torch.from_numpy(img_a), torch.from_numpy(img_b))
    # print(torch.from_numpy(img_a).unsqueeze(0).permute(0, 3, 1, 2).size())
    I1 = cv2.imread(root1 + path1[i])
    I2 = cv2.imread(root2 + path2[i])
    # print(img_a.shape)
    I1 = torch.from_numpy(np.rollaxis(I1, 2)).float().unsqueeze(0) / 255.0
    I2 = torch.from_numpy(np.rollaxis(I2, 2)).float().unsqueeze(0) / 255.0
    I1 = Variable(I1, requires_grad=True)
    I2 = Variable(I2, requires_grad=True)

    # default constants
    K = [0.01, 0.035]
    L = 255
    window_size = 11

    ssim_value = ssim(I1, I2, K, window_size, L)
    ae_value = ae(I1,I2).detach().numpy()
    print(psnr_num.numpy())
    print(I1.shape)

    # ssim_num = ssim(torch.from_numpy(img_a1).unsqueeze(0).permute(0, 3, 1, 2).float(), torch.from_numpy(img_b1).unsqueeze(0).permute(0, 3, 1, 2).float())

    list_ssim.append(ssim_value.data)
    list_psnr.append(psnr_num)
    list_ae.append(ae_value)
print("mean psnr:", np.mean(list_psnr))  # ,list_psnr)
print("mean SSIM:", np.mean(list_ssim))  # ,list_ssim)
print("mean AE:", np.mean(list_ae))  # ,list_ssim)

# elapsed = (time.clock() - start)
# print("Time used:", elapsed)