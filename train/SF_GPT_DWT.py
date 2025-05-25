import torch
import torch.nn as nn
import torch.nn.functional as F

import common
import GPG
from network_module import *

# ---------------------------- Model Builder ---------------------------- #
def build_model(args):
    return SF_GPT()

# ---------------------- Weight Initialization -------------------------- #
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print(f'Initializing network with {init_type} type')
    net.apply(init_func)

# ---------------------- Blockify / Unblockify -------------------------- #
def blockify(image, n_blocks, block_size):
    '''Convert image to non-overlapping blocks: [B, C, H, W] -> [B*n_blocks, block_size, block_size]'''
    return F.unfold(image, kernel_size=block_size, stride=block_size).permute(0, 2, 1).reshape(-1, n_blocks, block_size, block_size)

def unblockify(image_block, img_size, n_blocks, block_size):
    '''Reconstruct image from blocks'''
    return F.fold(image_block.reshape(-1, n_blocks, block_size**2).permute(0, 2, 1),
                  output_size=(img_size, img_size), kernel_size=block_size, stride=block_size)

# ------------------------ Wavelet Transforms --------------------------- #
def dwt_init(x):
    x01, x02 = x[:, :, 0::2, :] / 2, x[:, :, 1::2, :] / 2
    x1, x2 = x01[:, :, :, 0::2], x02[:, :, :, 0::2]
    x3, x4 = x01[:, :, :, 1::2], x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), dim=1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_channel = in_channel // (r * r)
    out_height, out_width = r * in_height, r * in_width

    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:2*out_channel, :, :] / 2
    x3 = x[:, 2*out_channel:3*out_channel, :, :] / 2
    x4 = x[:, 3*out_channel:4*out_channel, :, :] / 2

    h = torch.zeros([in_batch, out_channel, out_height, out_width], device=x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def forward(self, x):
        return iwt_init(x)

# ------------------------ Fusion Block ----------------------------- #
class SF_GPG(nn.Module):
    def __init__(self, dim_imgfeat, dim_dwtfeat, kernel_size=3):
        super().__init__()
        self.conv_img = nn.Sequential(
            common.ResBlock(dim_imgfeat, kernel_size),
            common.ResBlockAttn(dim_imgfeat, kernel_size)
        )
        self.conv_dwt = nn.Sequential(
            common.ResBlock(dim_dwtfeat, kernel_size),
            common.ResBlockAttn(dim_dwtfeat, kernel_size)
        )
        self.SF_GPG = GPG.SF_GPG(
            dim_in=2 * dim_imgfeat, dim_head=dim_imgfeat,
            dim_out=dim_imgfeat, heads=1, num_blocks=1
        )

    def forward(self, in_pix, in_dwt):
        out_pix = self.conv_img(in_pix)
        out_dwt = self.conv_dwt(in_dwt)
        fused = self.SF_GPG(out_pix, out_dwt)
        return fused, out_dwt

# ---------------------- Main Model ------------------------------- #
class SF_GPT(nn.Module):
    def __init__(self):
        super().__init__()
        n_feats = 128
        kernel_size = 3
        in_channel_img = 4
        in_channel_dwt = 4
        out_channel = 12

        self.n_blocks = 4096
        self.block_size = 4
        self.patch_size = 256

        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.dwt = DWT()
        self.pix_shuffle = nn.PixelShuffle(2)

        # Head
        self.head1 = nn.Sequential(
            nn.Conv2d(in_channel_img, n_feats // 2, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats // 2),
            nn.Conv2d(n_feats // 2, n_feats, kernel_size, padding=kernel_size // 2)
        )
        self.head_dwt = nn.Sequential(
            nn.Conv2d(in_channel_dwt, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2)
        )

        # Body
        self.body = nn.ModuleList([
            SF_GPG(n_feats, n_feats, kernel_size) for _ in range(30)
        ])

        # Tail
        self.tail = common.default_conv(n_feats, out_channel, kernel_size)

    def forward(self, x):
        dwt_input = self.dwt(x)                      # [B, 4, 128, 128]
        x_pix = self.head1(dwt_input)                # [B, 128, 128, 128]
        x_dwt = self.head_dwt(dwt_input)             # [B, 128, 128, 128]

        for i, layer in enumerate(self.body):
            if i == 0:
                res_pix, x_dwt = layer(x_pix, x_dwt)
            else:
                res_pix, x_dwt = layer(res_pix, x_dwt)

        out = self.tail(res_pix)                     # [B, 12, 128, 128]
        out = self.pix_shuffle(out)                  # [B, 3, 256, 256]
        return out

# ------------------------- Test Run ----------------------------- #
if __name__ == "__main__":
    net = SF_GPT()
    input_tensor = torch.randn(1, 1, 256, 256)
    output = net(input_tensor)
    print(f"Output shape: {output.shape}")  # Should print [1, 3, 256, 256]
