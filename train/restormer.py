import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class LKA_back_new_attn(nn.Module):
    def __init__(self, dim):
        super(LKA_back_new_attn,self).__init__()
        hidden = int(2*dim)
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)
        self.conv1_0 = nn.Conv2d(dim, dim, kernel_size=1)

        self.conv1_1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.SiLU()

        self.conv_spatial = nn.Conv2d(hidden, hidden, 7, stride=1, padding=3, groups=hidden)

        self.conv1_4 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.Conv11 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=(1, 11), padding=padding1, stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(11, 1), padding=padding2, stride=1,
                                              dilation=1, groups=hidden))
        self.Conv21 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),nn.Conv2d(hidden, hidden, kernel_size=(1, 21), padding=(0, int(21 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(21, 1), padding=(int(21 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))

        self.Conv31 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),
                                    nn.Conv2d(hidden, hidden, kernel_size=(1, 31), padding=(0, int(31 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(31, 1), padding=(int(31 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))

        self.conv1_5 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # # print(f"   lka input {x.shape}")
        u = self.conv1_0(x)

        attn = self.conv1_1(x)
        attn = self.act(attn)
        attn = self.conv_spatial(attn)


        u3 = attn.clone()
        attn = self.conv1_4(attn)
        attn = self.Conv11(attn)
        attn = self.act(attn + u3)

        u4 = attn.clone()
        attn = self.Conv21(attn)
        attn = self.act(attn + u4)

        u5 = attn.clone()
        attn = self.Conv31(attn)
        attn = self.act(attn + u5)

        # u6 = attn.clone()
        # attn = self.Conv41(attn)
        # attn = self.act(attn + u6)

        attn = self.conv1_5(attn)

        out1 = u * attn
        out1 = self.proj_1(out1)
        # print(f"   lka output {out1.shape}")

        return out1


       



# 使用哈尔 haar 小波变换来实现二维离散小波
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # h = torch.zeros([out_batch, out_channel, out_height,
    #                  out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x).cuda()
    
class ConvFFN_1(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias', bias=False):
        super(ConvFFN_1, self).__init__()
        self.conv1_1 = nn.Conv2d(dim , dim * 4, kernel_size=1, bias=bias)
        self.dwconv3_1 = nn.Conv2d(dim*4 , dim * 4, kernel_size=3, padding=1, groups=int(dim *4),bias=bias)  # depthwise conv
        self.act = nn.SiLU(inplace=True)
        self.tras_conv1 = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.dwconv3_1(out1)
        out1 = self.act(out1)
        out1 = self.tras_conv1(out1)
        return out1


##### ParaHybrid block
#### 该模块也有效 32.75dB 1000epochs
class ParaHybridBlock(nn.Module):
    def __init__(self, dim, head=4,ffn_expansion_factor=4, distillation_rate=0.25, LayerNorm_type='WithBias',bias=False):
        super(ParaHybridBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        # self.attn = inceptionAttn(dim)
        # self.attn = inceptionAttn_back(dim)
        self.attn = LKA_back_new_attn(dim)

        # self.largeKernelB = LargeKernelB(dim)
        # self.transformerB = TransformerBlock(dim,num_heads=head)
        # # self.ConvTransformerT = ConvformerBlock(dim)
        #
        # self.conv1_1 = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=True)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = ConvFFN_1(dim)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = ConvFFN_new(dim)
        # self.ffn = ConvFFN_new(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
##---------- Restormer -----------------------
###### dim可以调整为64
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=1,
                 dim=4,
                 num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 img_range=1.,
                 upscale=4,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()
        self.img_range = img_range
        self.upscale = upscale
        num_feat = 64
        if inp_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)


        self.encoder_level1 = nn.Sequential(*[ParaHybridBlock(dim,heads[i]) for i in range(num_blocks[0])])

        # self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        # self.encoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 1),heads[i]) for i in range(num_blocks[1])])

        self.down1_2 = DWT()
        self.encoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2),heads[i]) for i in range(num_blocks[1])])

        # self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        # self.encoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2),heads[i]) for i in range(num_blocks[2])])
        self.down2_3 = DWT()  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 4), heads[i]) for i in range(num_blocks[2])])

        # self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        # self.latent = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 3),heads[i]) for i in range(num_blocks[3])])
        self.down3_4 = DWT()  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 6), heads[i]) for i in range(num_blocks[3])])

        # self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        # self.decoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2),heads[i]) for i in range(num_blocks[2])])
        self.up4_3 = IWT()  ## From Level 4 to Level 3
        self.decoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 4), heads[i]) for i in range(num_blocks[2])])

        # self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        # self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 1),heads[i]) for i in range(num_blocks[1])])
        self.up3_2 = IWT()  ## From Level 3 to Level 2
        self.decoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2), heads[i]) for i in range(num_blocks[1])])

        # self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        # self.decoder_level1 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 1),heads[i]) for i in range(num_blocks[0])])
        self.up2_1 = IWT()  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[ParaHybridBlock(int(dim), heads[i]) for i in range(num_blocks[0])])

        #### 对LR输入的处理 ###########################
        self.LR_act = torch.nn.SiLU(inplace=True)  # SiLU

        self.refinement = nn.Sequential(*[ParaHybridBlock(int(dim),heads[i]) for i in range(num_refinement_blocks)])

        # for classical SR
        # self.conv_before_upsample = nn.Sequential(nn.Conv2d(int(dim), num_feat, 3, 1, 1),
        #                                           nn.SiLU(inplace=True))
        # self.upsample = Upsample_SR(upscale, num_feat)
        # self.conv_last = nn.Conv2d(dim, out_channels, 3, 1, 1)
        ##########################################################

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.input_upscale = 8

    def forward(self, rgb,in_thermal):
        in_thermal = F.interpolate(in_thermal, scale_factor= self.input_upscale, mode='bicubic',align_corners= False)# [2, 1, 256, 256]

        in_thermal = torch.clamp(in_thermal,0,1)
        # print(f'in_thermal {in_thermal.shape}')

        x = torch.cat((rgb,in_thermal),dim = 1 ) # [2, 4, 256, 256]
        ### unet直接处理LR图片
        # inp_enc_level1 = self.patch_embed(inp_img)
        ### unet先处理LR上采样两倍的图片
        out_enc_level1 = self.encoder_level1(x)
        print(f"out_enc_level1 {out_enc_level1.shape}")

        inp_enc_level2 = self.down1_2(out_enc_level1)
        #
        print(f"inp_enc_level2 {inp_enc_level2.shape}")

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        print(f"inp_enc_level2 {inp_enc_level2.shape}")

        inp_enc_level3 = self.down2_3(out_enc_level2)
        print(f"inp_enc_level3 {inp_enc_level3.shape}")

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        print(f"out_enc_level3 {out_enc_level3.shape}")


        inp_enc_level4 = self.down3_4(out_enc_level3)
        print(f"inp_enc_level4 {inp_enc_level4.shape}")

        latent = self.latent(inp_enc_level4)
        print(f"latent {latent.shape}")
        ### 小波变换
        inp_dec_level3 = self.up4_3(latent) + out_enc_level3
        print(f"inp_dec_level3 {inp_dec_level3.shape}")


        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        print(f"out_dec_level3 {out_dec_level3.shape}")



        ############## ori
        ### 小波变换
        inp_dec_level2 = self.up3_2(out_dec_level3) + out_enc_level2
        print(f"inp_dec_level2 {inp_dec_level2.shape}")


        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        print(f"out_dec_level2 {out_dec_level2.shape}")

        ##### 解码层，融合编码层对应的多尺度特征 l1 and l2
        # fuse_l1andl2 = self.fuse_multiple_l1andl2(out_enc_level1,out_enc_level2)

        ############################

        #### ori
        ### 小波变换
        inp_dec_level1 = self.up2_1(out_dec_level2) + out_enc_level1
        print(f"inp_dec_level1 {inp_dec_level1.shape}")
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        print(f"out_dec_level1 {out_dec_level1.shape}")

        ######## 在最后进行上采样 x2 -> refine -> x2 -> refine
        # out_dec_level1 = self.upsample_1(out_dec_level1)
        #################
        out_dec_level1 = self.refinement(out_dec_level1)
        print(f"out_dec_level1 {out_dec_level1.shape}")

        ##### 在最后再上采样
        out_dec_level1 = self.output(out_dec_level1)  #+ x_up
        print(f"out_dec_level1 {out_dec_level1.shape}")


        return out_dec_level1
    

# if __name__ == '__main__':

#     a = torch.randn(1,3,256,256)
#     a_1 = torch.randn(1,1,32,32)
#     # model = Restormer(dim=64)
#     model = Restormer()
#     out = model(a,a_1)
#     print('out:', out.shape)