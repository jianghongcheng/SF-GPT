import argparse
import os

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # pre-train, saving, and loading parameters
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 10, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--save_path', type = str, default = './models', help = 'save the pre-trained model to certain path')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'save the pre-trained model to certain path')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    #/home/hjq44/Documents/3/nir/train/models/SCGAN_noGAN_epoch2000_bs4.pth
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'True for more than 1 GPU, we recommend to use 4 NVIDIA A6000 GPUs')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # training parameters
    parser.add_argument('--epochs', type = int, default = 500, help = 'number of epochs of training') # change if fine-tune
    parser.add_argument('--batch_size', type = int, default = 8, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--gan_mode', type = str, default = 'noGAN', help = 'type of GAN: [noGAN]')
    parser.add_argument('--network_mode', type = str, default = 'dwt', help = 'type of network: [dct,dwt,cfat,hat,han,hma,edsr,grl,swinir,], mst is recommended')



    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                        help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0,
                        help='weight decay')


    # VCIP dataset
    parser.add_argument('--baseroot_rgb', type = str, default = './dataset/VCIP_20_aug/train_rgb', help = 'color image baseroot')#keep same name
    parser.add_argument('--baseroot_gray', type = str, default = './dataset/VCIP_20_aug/train_nir', help = 'nir image baseroot')####ÃÂ»ÃÂ nir
    parser.add_argument('--baseroot_rgb_test', type = str, default = './dataset/VCIP_20_aug/test_nir', help = 'color image baseroot')#keep same name
    parser.add_argument('--baseroot_gray_test', type = str, default = './dataset/VCIP_20_aug/test_rgb', help = 'nir image baseroot')####ÃÂÃÂ»ÃÂÃÂ nir
    # ARRI dataset

    
    parser.add_argument('--smaller_coeff', type = int, default = 1, help = 'sample images')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    import trainer
    # # import trainer_l1 as trainer
    # import trainer_ssim as trainer

    trainer.trainer_noGAN(opt)

