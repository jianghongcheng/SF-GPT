import common_edsr as common
# import attention
import torch.nn as nn
import torch.nn.functional as F

import torch
def make_model():

        return EDSR()

class EDSR(nn.Module):
    def __init__(self,  conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblock = 32
        n_feats = 256
        kernel_size = 3 
        scale = 1 # 2 4
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(1, rgb_mean, rgb_std)
        self.first  = nn.Conv2d(4, 1, kernel_size=3, padding=1)

        #self.msa = attention.PyramidAttention(channel=256, reduction=8,res_scale=args.res_scale);         
        # define head module
        m_head = [conv(1, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblock//2)
        ]
        #m_body.append(self.msa)
        for _ in range(n_resblock//2):
            m_body.append( common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, 3, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(1, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):


        
        #  output_rgb = F.interpolate(x_rgb, scale_factor=0.0625, mode='bilinear', align_corners=False)
        # out = torch.cat([x, output_rgb], 1)
        #  x = self.first(out)
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

# if __name__ == '__main__':
#     input = torch.rand(4, 1,32, 32)
#     input_rgb = torch.rand(4, 3, 256, 256)

#     model = EDSR()
#     out = model(
#                 input)
#     # print(model) out.shape
#     print(out.shape)
# #     # device = torch.device('cuda:0')
# #     # input = input.to(device)
# #     # model.eval()
# #     # model = model.to(device)
# #     # floaps, params = profile(model, inputs=(input,))
# #     # print('floaps: ', floaps)
# #     # print('params: ', params)
