'''
build model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from maml.net_utils import convBnRelu, basicBlock, bottleNeck
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class decoderBlock(nn.Module):
    '''
    build decoder block in ETnet
    '''

    def __init__(self, inc, outc, unic, is_up=True, crop=False):
        '''
        #parameters:
        @inc: input channel from previous layer
        @outc: channel of feature maps from encoder layer
        @unic: output channel of decoder block
        @is_up: if upsample input feature maps
        @crop: if crop upsampled feature maps
        '''

        super(decoderBlock, self).__init__()
        super(decoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(inc, outc, kernel_size=1, bias=False)
        self.deformcon1 = DeformConv2d(outc, outc, kernel_size=3, padding=1, stride=1,).cuda()
        self.bn1 = nn.BatchNorm2d(outc)

        self.conv2 = nn.Conv2d(outc, outc, kernel_size=1, bias=False)
        self.deformcon2 = DeformConv2d(outc, outc, kernel_size=3, padding=1, stride=1,).cuda()
        self.bn2 = nn.BatchNorm2d(outc) 
        
        self.conv3 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1, bias=False)
        self.deformcon3 = DeformConv2d(outc, outc, kernel_size=3, padding=1, stride=1,).cuda()
        self.bn3 = nn.BatchNorm2d(outc)
        
        self.conv4 = nn.Conv2d(outc, unic, kernel_size=1, bias=False)
        self.deformcon4 = DeformConv2d(unic, unic, kernel_size=3, padding=1, stride=1,).cuda()
        self.bn4 = nn.BatchNorm2d(unic)

        self.relu = nn.ReLU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.is_up = is_up
        self.crop = crop

    def forward(self, x1, x2):
        '''
        #parameters:
        @x1: input from previous layer
        @x2: input from encoder layer
        '''
#         x1 = self.relu(self.bn1(self.deformcon1(self.conv1(x1))))
        x1 = self.relu(self.bn1(self.conv1(x1)))

        if self.is_up == True:
            x1 = self.up(x1)

        if self.crop == True:
            x1 = x1[:,:,:-1,:-1]

        out = x1+x2
         
#         out = self.relu(self.bn2(self.deformcon2(self.conv2(out))))
#         out = self.relu(self.bn3(self.deformcon3(self.conv3(out))))
#         out = self.relu(self.bn4(self.deformcon4(self.conv4(out))))
        
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))

        return out

################   Spatial Attention Module  ##################

# class SMBasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(SMBasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

# class ChannelPool(nn.Module):
#     def forward(self, inc):
#         return torch.cat( (torch.max(inc,1)[0].unsqueeze(1), torch.mean(inc,1).unsqueeze(1)), dim=1 )

# class SM(nn.Module):
#     def __init__(self):
#         super(SM, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = SMBasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = F.sigmoid(x_out) # broadcasting
#         return x * scale

################   Spatial Attention Module End  ##################

class WB(nn.Module):
    '''
    a weight block
    '''
    
    def __init__(self, inc, outc, reduction_ratio=16):
        super(WB, self).__init__()

        self.basic_conv1 = convBnRelu(inc, outc, 3, p=1)
        self.w1= nn.Sequential(
            nn.Conv2d(outc, outc//reduction_ratio, 1),
            nn.ReLU()
            )
        self.w2 = nn.Sequential(
            nn.Conv2d(outc//reduction_ratio, outc, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid() 
#         self.sm = SM()

    def forward(self, x):
        out = self.basic_conv1(x)

        attention1 = self.w2(self.w1(self.avgpool(out)))     
        attention2 = self.w2(self.w1(self.maxpool(out)))

        fout = out * self.sigmoid(attention1 + attention2)
#         fout = self.sm(fout)    # add spatail attention at each channel attention block(total 3 spatail and 3 channel block)

        return fout

class AM(nn.Module):
    '''
    aggregation module
    '''

    def __init__(self, inc1, inc2, inc3, outc=128):
        super(AM, self).__init__()

        self.weight_block1 = WB(inc1, outc)
        self.weight_block2 = WB(inc2, outc)
        self.weight_block3 = WB(inc3, outc)

        #self.weight_block12 = WB(outc, outc)
        self.conv1x1_1 = convBnRelu(outc, outc, 1)
        self.conv1x1_2 = convBnRelu(outc, outc, 1)

        self.seg_conv3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),     #500 * 500
            nn.Conv2d(outc, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),     #1000 * 1000
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        

    def forward(self, x1, x2, x3):
        out1 = self.weight_block1(x1)          #128, 64 * 64
        out2 = self.weight_block2(x2)          #128, 128 * 128
        out3 = self.weight_block3(x3)          #128, 256 * 256

        out1 = self.up(out1)
        #out1 = out1[:,:,:-1,:-1]                               #128, 125 * 125
        out12 = out1+out2                                      #128, 125 * 125
        out12 = self.conv1x1_1(out12)

        out123 = self.up(out12) + out3                        #128, 250 * 250
        out123 = self.conv1x1_2(out123)

        seg_out = self.seg_conv3(out123)

        return seg_out


                          #####ASPP##########
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            # DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1,).cuda(),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
        
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # DeformConv2d(out_channels, out_channels, kernel_size=1, padding=1, stride=1,).cuda(),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 1024   #modify this parameter to fit the image channel
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x)) 
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class SegNet(nn.Module):

    def __init__(self, inc, block, device, nums=[3, 4, 6, 3]):
        seg_outc = 128

        super(SegNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, inc, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(inc),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder2 = self._make_layer(inc, inc, block, nums[0], downsample=False)
        self.encoder3 = self._make_layer(4*inc, 2*inc, block, nums[1])
        self.encoder4 = self._make_layer(8*inc, 4*inc, block, nums[2])
        self.encoder5 = self._make_layer(16*inc, 8*inc, block, nums[3], downsample=False, dilation=2)   #using ASPP instead

        ##add ASPP block
        # self.aspp = ASPP(16*inc, [6,12,18])  

        self.decoder1 = decoderBlock(32*inc, 16*inc, 16*inc, False)
        self.decoder2 = decoderBlock(16*inc, 8*inc, 8*inc, True)
        self.decoder3 = decoderBlock(8*inc, 4*inc, 4*inc)

        self.am = AM(16*inc, 8*inc, 4*inc)

        #self.sobel = sobelFilter(device)

    def forward(self, x):
        
        #################################
        #----------Encoder layer--------#
        #################################        
        en1_out = self.encoder1(x)                      #32 * 500, 500

        en2_out = self.encoder2(self.maxpool(en1_out))  #128, 250, 250
        en3_out = self.encoder3(en2_out)                #256, 125, 125
        en4_out = self.encoder4(en3_out)                #512, 63, 63
        en5_out = self.encoder5(en4_out)                #1024, 63, 63  ##using ASPP instead this layer

        #################################
        #--------------ASPP-------------#
        ################################# 
   
        # aspp_output = self.aspp(en4_out)
        
        #################################
        #----------Decoder layer--------#
        #################################    
        # de1_out = self.decoder1(aspp_output, en4_out)       
        de1_out = self.decoder1(en5_out, en4_out)      #512, 63, 63    ##using ASPP instead this layer
        de2_out = self.decoder2(de1_out, en3_out)      #256, 125, 125
        de3_out = self.decoder3(de2_out, en2_out)      #128, 250, 250

        #################################
        #------aggregation module-------#
        ################################# 
        seg_out = self.am(de1_out, de2_out, de3_out)  

        return seg_out

    def _make_layer(self, inc, outc, block, nums, downsample=True, dilation=1):
        '''
        #inc: input channel from previous layer
        #outc: output channel of this block
        '''
        layers = []

        if downsample == True:
            my_downsample = nn.Sequential(
                nn.Conv2d(inc, outc*4, 1, stride=2, bias=False),   
                nn.BatchNorm2d(outc*4)
            )
            layers.append(block(inc, outc, stride=2, skip=my_downsample))

        else:
            my_downsample = nn.Sequential(
                nn.Conv2d(inc, outc*4, 1, stride=1, bias=False),  
                nn.BatchNorm2d(outc*4)
            )
            layers.append(block(inc, outc, skip=my_downsample))
        
        for _ in range(nums-1):
            layers.append(block(outc*4, outc, dilation=dilation))

        return nn.Sequential(*layers)

def build_network(device, inc=32):
    return SegNet(inc, bottleNeck, device) 
# def build_network(device, inc=32, is_1000=False):
#     return SegNet(inc, bottleNeck, device, is_1000=is_1000)