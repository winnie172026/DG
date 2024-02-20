import torch
import torch.nn as nn
import torch.nn.functional as F 

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

class convBnRelu(nn.Module):
    '''
    conv2d -> batch normalizatoin -> relu
    
    Parameters
    ----------
    @inc: input channel
    @outc: output channel
    @k: kernel_size
    @s: stride
    @p: padding
    '''

    def __init__(self, inc, outc, k, s=1, p=0):
        super(convBnRelu, self).__init__()
        self.module = nn.Sequential(
                    nn.Conv2d(inc, outc, k, s, padding=p, bias=False),
                    nn.BatchNorm2d(outc),
                    nn.ReLU())
    
    def forward(self, x):

        return self.module(x)


class basicBlock(nn.Module):
    '''
    build basic block in residual net, 2 layers
    
    Parameters
    ----------
    @inc: input channel
    @outc: output channel
    @stride: stride
    @dilation: dilation value
    @skip: skip connection method
    '''
    
    def __init__(self, inc, outc, stride=1, dilation=1, skip=None):
        super(basicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, 3, stride=stride, padding=dilation, \
                                dilation= dilation, bias=False)
        self.conv_offset2d1 = DeformConv2d(outc, outc, kernel_size=3, padding=1, stride=1,).cuda()
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, 3, padding=1, bias=False)
        self.conv_offset2d2 = DeformConv2d(outc, outc, kernel_size=3, padding=1, stride=1,).cuda() 
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU()

        self.skip = skip

    def forward(self, x):
        shortcut = x

        out = self.relu(self.bn1(self.conv_offset2d1(self.conv1(x))))
        out = self.bn2(self.conv_offset2d2(self.conv2(out)))
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))

        if self.skip is not None:
            shortcut = self.skip(x)

        out += shortcut
        out = self.relu(out)

        return out
        
class bottleNeck(nn.Module):
    '''
    build bottle_neck in residual net, 3 layers
    
    Parameters
    ----------
    @inc: input channel
    @outc: output channel
    @stride: stride
    @dilation: dilation value
    @skip: skip connection method
    '''

    def __init__(self, inc, outc, stride=1, dilation=1, skip=None):
        super(bottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=1, bias=False)
        # self.conv_offset2d1 = DeformConv2d(outc, outc, kernel_size=3, padding=1, stride=1,).cuda()
        self.bn1 = nn.BatchNorm2d(outc)
        
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, stride=stride, padding=dilation, \
                                            dilation=dilation, bias=False)
        # self.conv_offset2d2 = DeformConv2d(outc, outc, kernel_size=3, padding=1, stride=1,).cuda()                                      
        self.bn2 = nn.BatchNorm2d(outc)
        
        self.conv3 = nn.Conv2d(outc, outc*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outc*4)

        self.relu = nn.ReLU()
        self.skip = skip

    def forward(self, x):
        
        shortcut = x
        
        # out = self.relu(self.bn1(self.conv_offset2d1(self.conv1(x))))
        # out = self.relu(self.bn2(self.conv_offset2d2(self.conv2(out))))
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            shortcut = self.skip(x)

        out += shortcut
        out = self.relu(out)

        return out
        

class sobelFilter(nn.Module):
    '''
    apply sobel filter on input tensor
    output tensor with 2 channels
    '''

    def __init__(self, device):
        super(sobelFilter, self).__init__()

        self.sobel_x = torch.tensor([[1., 0., -1.], \
                            [2., 0., -2.], \
                            [1., 0., -1.]]).unsqueeze(0).unsqueeze(0).to(device)

        self.sobel_y = torch.tensor([[1.,   2.,  1.],\
                            [0.,   0.,  0.],\
                            [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, x):
        grad_out = F.pad(x, (1, 1, 1, 1), 'reflect')
        grad_out_x = F.conv2d(grad_out, self.sobel_x)
        grad_out_y = F.conv2d(grad_out, self.sobel_y)
        grad_out = torch.cat((grad_out_x, grad_out_y), 1)

        return grad_out