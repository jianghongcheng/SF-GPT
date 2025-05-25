import torch
import torch.nn as nn
import torch.nn.functional as F

import common
import GPG
from network_module import *
from torch_dct import dct_2d

def build_model(args):
    return SF_GPT()

# ---------------------------- Weight Initialization ---------------------------- #
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

# ---------------------------- Blockify / Unblockify ---------------------------- #
def blockify(image, n_blocks, block_size):
    return F.unfold(image, kernel_size=block_size, stride=block_size).permute(0,2,1).reshape(-1, n_blocks, block_size, block_size)

def unblockify(image_block, img_size, n_blocks, block_size):
    return F.fold(image_block.reshape(-1, n_blocks, block_size**2).permute(0, 2, 1), output_size=(img_size, img_size), kernel_size=block_size, stride=block_size)

# ---------------------------- Fusion Block ---------------------------- #
class SF_GPTFusionBlock(nn.Module):
    def __init__(self, dim_imgfeat, dim_dctfeat, kernel_size=3):
        super().__init__()
        self.conv_img = nn.Sequential(
            common.ResBlock(dim_imgfeat, kernel_size),
            common.ResBlockAttn(dim_imgfeat, kernel_size)
        )
        self.conv_dct = nn.Sequential(
            common.ResBlock(dim_dctfeat, kernel_size),
            common.ResBlockAttn(dim_dctfeat, kernel_size)
        )
        self.stage_tconv = nn.ConvTranspose2d(dim_dctfeat, dim_imgfeat, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.fusion = GPG.SF_GPG(dim_in=2 * dim_imgfeat, dim_head=dim_imgfeat, dim_out=dim_imgfeat, heads=1, num_blocks=1)

    def forward(self, in_pix, in_dct):
        out_pix = self.conv_img(in_pix)
        out_dct = self.conv_dct(in_dct)
        out_pix = self.fusion(out_pix, self.stage_tconv(out_dct, output_size=in_pix.shape[2:]))
        return out_pix, out_dct

# ---------------------------- Main Network ---------------------------- #
class SF_GPT(nn.Module):
    def __init__(self):
        super().__init__()
        n_feats = 128
        kernel_size = 3
        in_channel_img = 4
        in_channel_dct = 16
        out_channel = 12

        self.n_blocks, self.block_size = 4096, 4
        self.patch_size = 256
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.pix_shuffle = nn.PixelShuffle(2)
        n_basicblock = 10

        # Head
        self.head1 = nn.Sequential(
            nn.Conv2d(in_channel_img, n_feats // 2, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats // 2),
            nn.Conv2d(n_feats // 2, n_feats, kernel_size, padding=kernel_size // 2)
        )
        self.head_dct = nn.Sequential(
            nn.Conv2d(in_channel_dct, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats, 2 * n_feats, kernel_size, padding=kernel_size // 2)
        )

        # Body
        self.body = nn.ModuleList([SF_GPTFusionBlock(n_feats, 2 * n_feats, kernel_size) for _ in range(n_basicblock)])

        # Tail
        self.tail = common.default_conv(n_feats, out_channel, kernel_size)

    def forward(self, x):
        # Step 1: DCT blockify
        img_block = blockify(x[:, 0:1, :, :], self.n_blocks, self.block_size)
        dct_block = dct_2d(img_block, norm='ortho')
        dct_nir = unblockify(dct_block[:, 0:self.n_blocks], self.patch_size, self.n_blocks, self.block_size)
        dct_input = F.pixel_unshuffle(dct_nir, self.block_size)  # [B, 16, 64, 64]
        unshuffled_x = self.pixel_unshuffle(x)                   # [B, 4, 128, 128]

        # Step 2: Encode
        feat_img = self.head1(unshuffled_x)
        feat_dct = self.head_dct(dct_input)

        # Step 3: Fusion through SF_GPTFusionBlock
        for i, layer in enumerate(self.body):
            if i == 0:
                res, feat_dct = layer(feat_img, feat_dct)
            else:
                res, feat_dct = layer(res, feat_dct)

        # Step 4: Decode
        res = self.tail(res)
        out = self.pix_shuffle(res)
        return out

# ---------------------------- Test ---------------------------- #
if __name__ == "__main__":
    net = SF_GPT()
    net.eval()
    input_tensor = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        output = net(input_tensor)
    print(f"Output shape: {output.shape}")  # Expect [1, 3, 256, 256]
