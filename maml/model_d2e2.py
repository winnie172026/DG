import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(self.in_channels))
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, self.max_pool_layer)
            self.phi = nn.Sequential(self.phi, self.max_pool_layer)

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# class EdgeConv2d(nn.Module):
#     """docstring for EdgeConv2d"""
#     def __init__(self, in_channels, out_channels):
#         super(EdgeConv2d, self).__init__()
#         ## 用nn.conv2d定义卷积操作
#         self.conv_op = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         print("conv_op: ", self.conv_op.weight.data.shape)
#         ## 卷积核
#         # sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') #/ 3  ## outline
#         # sobel_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype='float32')   ## emboss
#         sobel_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype='float32') / 3  ## sharpen
 
#         sobel_kernel = sobel_kernel.reshape(1, 1, 3, 3)
#         print('sobel_kernel 1: ', sobel_kernel.shape)
#         ## 输出通道数为3
#         sobel_kernel = np.repeat(sobel_kernel, out_channels, axis=0)
#         print('sobel_kernel 2: ', sobel_kernel.shape)
#         ## 输入通道数为3
#         sobel_kernel = np.repeat(sobel_kernel, in_channels, axis=1)
#         print('sobel_kernel 3: ', sobel_kernel.shape)
#         self.conv_op.weight.data = torch.from_numpy(sobel_kernel)

#     def forward(self, input):
#         """description_method """
#         return self.conv_op(input)


class Sobel_filter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sobel_filter, self).__init__()
        self.conv_op_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv_op_y = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # print("conv_op: ", self.conv_op_x.weight.data.shape)

        Gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) 
        Gy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])

        Gx = Gx.view(size=(1, 1, 3, 3))
        Gy = Gy.view(size=(1, 1, 3, 3))

        sobel_Gx = Gx.repeat(out_channels, in_channels, 1, 1)
        sobel_Gy = Gy.repeat(out_channels, in_channels, 1, 1)
        self.conv_op_x.weight = nn.Parameter(sobel_Gx, requires_grad=False)
        self.conv_op_y.weight = nn.Parameter(sobel_Gy, requires_grad=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = nn.Parameter(torch.tensor([2.0], device=device))
        self.beta = nn.Parameter(torch.tensor([1.0], device=device))
        

        self.lambda_min = torch.tensor(0.2).to(device)
        self.lambda_max = torch.tensor(0.7).to(device)
        self.con = torch.tensor(1.0).to(device)


    def forward(self, input):
        
        gx = self.conv_op_x(input)
        gy = self.conv_op_y(input)
        # print("gx: ", gx)
        # print("gy: ", gy)
        
        # gx2 = torch.matmul(gx, gx)
        # gy2 = torch.matmul(gy, gy)
        gx2 = torch.mul(gx, gx)
        gy2 = torch.mul(gy, gy)

        x = torch.sqrt(gx2 + gy2)   ## edge map
        # print("x: ", x)

        x_temp = 1.0 - (x.to(device) - self.lambda_min) / (self.lambda_max - self.lambda_min)
        # print(x_temp.size(), type(x_temp))

        x_temp = x_temp * self.alpha + self.beta
        # print("x_temp: ", x_temp.size())
    
        # flag_temp = torch.where()
        
        x_final = torch.where((x.to(device) > self.lambda_min) & (x.to(device) < self.lambda_max), x_temp, self.con)
        # print("x_final: ", x_final.size())

        return x_final
        
class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """
    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, stride=2)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        x = self.up(x)
        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        x_skip = F.pad(x_skip, (0, W_diff, 0, H_diff), mode='reflect')
        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Transfer Learning ResNet as Encoder part of UNet
class ResUNet(nn.Module):
    def __init__(self, out_c, n_layers=34, pretrained=True, fixed_feature=False):
        super().__init__()
        # load weight of pre-trained resnet
        if n_layers == 18:
            self.resnet = models.resnet18(pretrained=pretrained)
            l = [64, 64, 128, 256, 512]
        elif n_layers == 34:
            self.resnet = models.resnet34(pretrained=pretrained)
            l = [64, 64, 128, 256, 512]
        elif n_layers == 50:
            self.resnet = models.resnet50(pretrained=pretrained)
            l = [64, 256, 512, 1024, 2048]
        elif n_layers == 101:
            self.resnet = models.resnet101(pretrained=pretrained)
            l = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError

        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # self.edgeDetect = EdgeConv2d(in_channels=3, out_channels=1)
        self.edgeDetect = Sobel_filter(in_channels=3, out_channels=1)
        self.denoise1 = NonLocalBlockND(in_channels=l[0], inter_channels=l[0])
        self.denoise2 = NonLocalBlockND(in_channels=l[1], inter_channels=l[1])
        self.denoise3 = NonLocalBlockND(in_channels=l[2], inter_channels=l[2])

        # up conv
        self.u5 = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6 = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7 = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8 = ConvUpBlock(l[1], l[0], dropout_rate=0.1)
        # final conv
        self.ce5 = nn.ConvTranspose2d(l[3], out_c, 2, stride=2)
        self.ce6 = nn.ConvTranspose2d(l[2], out_c, 2, stride=2)
        self.ce7 = nn.ConvTranspose2d(l[1], out_c, 2, stride=2)
        self.ce8 = nn.ConvTranspose2d(l[0], out_c, 2, stride=2) 

    def forward(self, x):
        # print("input: ", x.size())

        edge_im = x
        edge = self.edgeDetect(edge_im) 
        # print("edge: ", edge.size())

        x = self.resnet.conv1(x)
        # print("conv1: ", x.size())
        x = self.resnet.bn1(x)
        # print("bn1: ", x.size())
        x = c1 = self.resnet.relu(x)
        # print("relu: ", x.size())
        x = self.resnet.maxpool(x)
        # print("pool: ", x.size())


        denoise_c1 = self.denoise1(c1)
        edge1 = F.interpolate(edge, size=(x.size(2), x.size(3)), mode='nearest')
        x = edge1 * x

        x = c2 = self.resnet.layer1(x)
        # print("layer1: ", x.size())
        denoise_c2 = self.denoise2(c2)
        edge2 = F.interpolate(edge, size=(x.size(2), x.size(3)), mode='nearest')
        x = edge2 * x

        x = c3 = self.resnet.layer2(x)
        # print("layer2: ", x.size())
        denoise_c3 = self.denoise3(c3)
        edge3 = F.interpolate(edge, size=(x.size(2), x.size(3)), mode='nearest')
        x = edge3 * x

        x = c4 = self.resnet.layer3(x)
        # print("layer3: ", x.size())
        x = self.resnet.layer4(x)
        # print("layer4: ", x.size())
        x = self.u5(x, c4)
        # print("u5: ", x.size())
        out5 = self.ce5(x)
        # print("out5: ", out5.size())
        out5 = F.interpolate(out5, size=(edge_im.size(2), edge_im.size(3)), mode='nearest')
        # print("out5_after: ", out5.size())
        x = self.u6(x, denoise_c3)
        # print("u6: ", x.size())
        out6 = self.ce6(x)
        # print("out6: ", out6.size())
        out6 = F.interpolate(out6, size=(edge_im.size(2), edge_im.size(3)), mode='nearest')
        # print("out6_after: ", out6.size())
        x = self.u7(x, denoise_c2)
        # print("u7: ", x.size())
        out7 = self.ce7(x)
        # print("out7: ", out7.size())
        out7 = F.interpolate(out7, size=(edge_im.size(2), edge_im.size(3)), mode='nearest')
        # print("out7_after: ", out7.size())
        x = self.u8(x, denoise_c1)
        # print("u8: ", x.size())
        out8 = self.ce8(x)
        # print("ce: ", out8.size())
        out8 = torch.sigmoid(out8)
        return out5, out6, out7, out8


def create_model(model_name, out_c, pretrained):
    if 'ResUNet' in model_name:
        model = ResUNet(out_c, n_layers=int(model_name[7:]), pretrained=pretrained)
    else:
        raise NotImplementedError()

    return model


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input = torch.randn(2, 3, 224, 224).to(device)

    RUnet = ResUNet(out_c=1, n_layers=34).to(device)

    out = RUnet(input)

    # input = torch.randn(2, 3, 512, 512)
    # edgeDetect = EdgeConv2d(in_channels=3, out_channels=1)
    # out = edgeDetect(input)
    # print("out: ", out.size())

    # SobelNet = Sobel_filter(in_channels=3, out_channels=1)
    #
    # input = torch.randn(2, 3, 224, 224)
    # out = SobelNet(input)
    # print("out: ", out.size())
