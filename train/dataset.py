import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import utils



def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]

class ColorizationDataset(Dataset):
    def __init__(self, opt):
        # Note that:
        # 1. opt: all the options
        # 2. imglist: all the image names under "baseroot"
        self.opt = opt
        imglist = utils.get_jpgs(opt.baseroot_rgb)
        if opt.smaller_coeff > 1:
            imglist = self.create_sub_trainset(imglist, opt.smaller_coeff)
        self.imglist = imglist
        self.opt.augmentation = True
        # self.opt.crop_size = 256

    def create_sub_trainset(self, imglist, smaller_coeff):
        # Sample the target images
        namelist = []
        for i in range(len(imglist)):
            if i % smaller_coeff == 0:
                a = random.randint(0, smaller_coeff - 1) + i
                namelist.append(imglist[a])
        return namelist

    def __getitem__(self, index):
        # Path of one image
        imgname = self.imglist[index]
        imgpath = os.path.join(self.opt.baseroot_rgb, imgname)
        # salpath = os.path.join(self.opt.baseroot_sal, imgname)
        graypath = os.path.join(self.opt.baseroot_gray, imgname)#add grayscale image


        # Read the images######change gray to nir
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # RGB output image
        grayimg = cv2.imread(graypath, 0)                              # Grayscale input image
        grayimg = np.expand_dims(grayimg , axis= 2)
        # sal = cv2.imread(salpath, -1)                               # Saliency map output image
        
        # if self.opt.augmentation:
        #     # Randomly flip horizontally
        #     if random.random() > 0.5:
        #         img = np.fliplr(img)
        #         grayimg = np.fliplr(grayimg)

        #     # Randomly flip vertically
        #     if random.random() > 0.5:
        #         img = np.flipud(img)
        #         grayimg = np.flipud(grayimg)
        
        # img_in, img_tar = get_patch(grayimg, img, patch_size=256, scale=4)
        self.upscale = 1
        self.crop_size = 0
        if self.crop_size > 0:
            h,w,c = img.shape
            xx = random.randint(0, h-self.crop_size)
            yy = random.randint(0, w-self.crop_size)
            # print(img.shape)
            # print(grayimg.shape)
            img = img[xx*self.upscale:xx*self.upscale+self.crop_size*self.upscale,yy*self.upscale:yy*self.upscale+self.crop_size*self.upscale,:]
            grayimg = grayimg[xx*self.upscale:xx*self.upscale+self.crop_size*self.upscale,yy*self.upscale:yy*self.upscale+self.crop_size*self.upscale,:]
        self.data_aug = 0
        if self.data_aug > 0:
            grayimg, img = augment(grayimg, img)
        # print(img.shape)
        # print(grayimg.shape)

        # print(rgb.shape)
        # print(hr.shape)
        # print(img_tar.shape)
        
        # # Random cropping
        # if self.opt.crop_size > 0:
        #     h, w = img.shape[:2]
        #     rand_h = random.randint(0, h - self.opt.crop_size)
        #     rand_w = random.randint(0, w - self.opt.crop_size)
        #     img = img[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size, :]
        #     grayimg = grayimg[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size, :]
        
        # # Normalized to [-1, 1]
        grayimg = np.ascontiguousarray(grayimg, dtype = np.float32)
        # grayimg = (grayimg - 128.0) / 128.0
        grayimg = grayimg/255
        img = np.ascontiguousarray(img, dtype = np.float32)
        # img = (img - 128.0) / 128.0
        img = img/255
        # sal = np.ascontiguousarray(sal, dtype = np.float32)
        # sal = sal / 255.0



        # To PyTorch Tensor
        grayimg = torch.from_numpy(grayimg).permute(2, 0, 1).contiguous()
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()




        return grayimg, img
    
    def __len__(self):
        return len(self.imglist)

class ColorizationDataset_Val(Dataset):
    def __init__(self, opt):
        # Note that:
        # 1. opt: all the options
        # 2. imglist: all the image names under "baseroot"
        self.opt = opt
        self.opt.augmentation = False

        imglist = utils.get_jpgs(opt.baseroot_rgb)
        self.imglist = imglist

    def __getitem__(self, index):
        # Path of one image
        imgname = self.imglist[index]
        imgpath = os.path.join(self.opt.baseroot_rgb_test, imgname)
        grayimgpath = os.path.join(self.opt.baseroot_gray_test, imgname)

        # Read the images
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # RGB output image
        grayimg = cv2.imread(grayimgpath, 0)
        grayimg = np.expand_dims(grayimg, axis=2)

        if self.opt.augmentation:
            # Randomly flip horizontally
            if random.random() > 0.5:
                img = np.fliplr(img)
                grayimg = np.fliplr(grayimg)

            # Randomly flip vertically
            if random.random() > 0.5:
                img = np.flipud(img)
                grayimg = np.flipud(grayimg)
        # # Random cropping
        # if self.opt.crop_size > 0:
        #     h, w = img.shape[:2]
        #     rand_h = random.randint(0, h - self.opt.crop_size)
        #     rand_w = random.randint(0, w - self.opt.crop_size)
        #     img = img[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size, :]
        #     grayimg = grayimg[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size, :]

        # Normalized to [-1, 1]
        grayimg = np.ascontiguousarray(grayimg, dtype=np.float32)
        grayimg = grayimg / 255
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = img / 255

        # To PyTorch Tensor
        grayimg = torch.from_numpy(grayimg).permute(2, 0, 1).contiguous()
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        # Return the grayscale image, RGB image, and the image name
        return grayimg, img, imgname

    def __len__(self):
        return len(self.imglist)


class ColorizationDataset_Test(Dataset):
    def __init__(self, opt):
        # Note that:
        # 1. opt: all the options
        # 2. imglist: all the image names under "baseroot"
        self.opt = opt
        imglist = utils.get_jpgs(opt.baseroot_rgb)
        self.imglist = imglist

    def __getitem__(self, index):
        # Path of one image
        imgname = self.imglist[index]
        imgpath = os.path.join(self.opt.baseroot_rgb, imgname)
        grayimgpath = os.path.join(self.opt.baseroot_gray, imgname)
        testimgpath = os.path.join(self.opt.baseroot_test, imgname)

        # Read the images
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB output image
        # grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)             # Grayscale input image
        grayimg = cv2.imread(grayimgpath, 0)
        # grayimg = cv2.imread(grayimgpath)
        grayimg = np.expand_dims(grayimg, axis=2)

        testimg = cv2.imread(testimgpath)
        testimg = cv2.cvtColor(testimg, cv2.COLOR_BGR2RGB)  # RGB output image


        # # Random cropping
        # if self.opt.crop_size > 0:
        #     h, w = img.shape[:2]
        #     rand_h = random.randint(0, h - self.opt.crop_size)
        #     rand_w = random.randint(0, w - self.opt.crop_size)
        #     img = img[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size, :]
        #     grayimg = grayimg[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size, :]

        # Normalized to [-1, 1]
        grayimg = np.ascontiguousarray(grayimg, dtype=np.float32)
        grayimg = grayimg / 255
        # grayimg = (grayimg - 128.0) / 128.0
        img = np.ascontiguousarray(img, dtype=np.float32)
        # img = (img - 128.0) / 128.0
        img = img / 255
        testimg = np.ascontiguousarray(testimg, dtype=np.float32)
        # img = (img - 128.0) / 128.0
        testimg = testimg / 255



        # To PyTorch Tensor
        grayimg = torch.from_numpy(grayimg).permute(2, 0, 1).contiguous()
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        testimg = torch.from_numpy(testimg).permute(2, 0, 1).contiguous()

        return grayimg, img, testimg

    def __len__(self):
        return len(self.imglist)


# if __name__ == "__main__":
    
#     a = torch.randn(1, 3, 256, 256)
#     b = a[:, [0], :, :] * 0.299 + a[:, [1], :, :] * 0.587 + a[:, [2], :, :] * 0.114
#     b = torch.cat((b, b, b), 1)
#     print(b.shape)

#     c = torch.randn(1, 1, 256, 256)
#     d = a * c
#     print(d.shape)
