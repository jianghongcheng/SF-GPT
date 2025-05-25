import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dct_loss import dct_loss
import dataset
import utils
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs')
import pytorch_ssim as pytorchssim
import torch.optim.lr_scheduler as lrs
from idct_loss import idct_loss
from validation import PSNR
from my_loss import Loss

criterion_PSNR = PSNR()
#------------------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#------------------------------------------------------------------------------


def trainer_noGAN(opt):
    LR = 0
    L1 = 0

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.sample_path):
        os.makedirs(opt.sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_Loss = torch.nn.L1Loss().cuda()




    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()

    else:
        generator = generator.cuda()



    

    # Save the model
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == 'epoch':
            model_name = '4_%s_epoch%d_bs%d.pth' % (opt.gan_mode, epoch, opt.batch_size)
        if opt.save_mode == 'iter':
            model_name = 'SCGAN_%s_iter%d_bs%d.pth' % (opt.gan_mode, iteration, opt.batch_size)
        save_name = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                # if 1:
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ColorizationDataset(opt)#######check train set
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    valset = dataset.ColorizationDataset_Val(opt)
    dataloader_val = DataLoader(valset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)

    # Optimizers
    # optimizer = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2))
    optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.epsilon, weight_decay=opt.weight_decay, amsgrad=False)
    # scheduler = lrs.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=opt.gamma)
    criterion_MSE = torch.nn.MSELoss().cuda()
    criterion_SSIM = pytorchssim.SSIM(window_size=11)

    from torch.optim.lr_scheduler import CosineAnnealingLR

    scheduler = CosineAnnealingLR(optimizer,  T_max=opt.epochs*len(dataloader), eta_min=1e-6)
    # print(opt.epochs*len(dataloader))

    def valid(dataloader_val,model):
        model.eval()
        val_PSNR = AverageMeter()
        val_SSIM = AverageMeter()
        for i, (true_L, true_RGB) in enumerate(dataloader_val):

            with torch.no_grad():
                true_L = true_L.cuda()
                true_RGB = true_RGB.cuda()

                fake_all = model(true_L)
                psnr_val = criterion_PSNR(fake_all,true_RGB)
                ssim_val = criterion_SSIM(fake_all, true_RGB)

            val_PSNR.update(psnr_val.data)
            val_SSIM.update(ssim_val.data)
        return val_PSNR.avg, val_SSIM.avg



    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    # ------------------------------------------

    track_l1loss_tensorboard = AverageMeter()
    track_ssimloss_tensorboard = AverageMeter()
    track_lr_tensorboard = AverageMeter()
    track_overallloss_tensorboard = AverageMeter()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_L, true_RGB) in enumerate(dataloader):#change data type

            # To device
            true_L = true_L.cuda()
            true_RGB = true_RGB.cuda()

            fake_all = generator(true_L)

            loss_L1 = criterion_Loss(fake_all, true_RGB) 
            # loss_mse = criterion_MSE(fake_all, true_RGB)
            loss_ssim = 1 - criterion_SSIM(fake_all, true_RGB)

            # Overall Loss and optimize
            loss = 10 * loss_L1 + 10 * loss_ssim


            track_l1loss_tensorboard.update(loss_L1.data)
            track_lr_tensorboard.update(optimizer.param_groups[0]['lr'] )
            track_ssimloss_tensorboard.update(loss_ssim.data)
            track_overallloss_tensorboard.update(loss.data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()


            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f]    [SSIM Loss: %.4f]  [LR: %.10f]  Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), loss_L1.item(),loss_ssim.item(),optimizer.param_groups[0]['lr'],time_left))

   

        writer.add_scalar('L1 loss',track_l1loss_tensorboard.avg,epoch+1)
        writer.add_scalar('SSIM loss',track_ssimloss_tensorboard.avg,epoch+1)
        writer.add_scalar('LR',track_lr_tensorboard.avg,epoch+1)
        writer.add_scalar('Total loss',track_overallloss_tensorboard.avg,epoch+1)
        save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            



