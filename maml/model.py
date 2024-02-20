import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

class pce(nn.Module):
    # parmid color embedding

    def __init__(self):
        super(pce, self).__init__()
        self.cma_4 = cma(256, 128)
        self.cma_3 = cma(128, 64)
        self.cma_2 = cma(64, 32)
        self.cma_1 = cma(32, 16)
        
        
    def forward(self, c, shortcuts):
        
        # change channels
        x_4_color, c_3 = self.cma_4(c, shortcuts[3]) # (b,128,h,w) 
        x_3_color, c_2 = self.cma_3(c_3, shortcuts[2]) # (b,64,h,w) 
        x_2_color, c_1 = self.cma_2(c_2, shortcuts[1]) # (b,32,h,w) 
        x_1_color, _ = self.cma_1(c_1, shortcuts[0]) # (b,16,h,w) 
        return [x_1_color, x_2_color, x_3_color, x_4_color]
        
class cma(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(cma, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode='nearest'))
        
    def forward(self, c, x):
        # x: gray image features 
        # c: color features

        # l1 distance
        channels = c.shape[1]
        sim_mat_l1 = -torch.abs(x-c) # <0  (b,c,h,w)
        sim_mat_l1 = torch.sum(sim_mat_l1, dim=1, keepdim=True) # (b,1,h,w)
        sim_mat_l1 = torch.sigmoid(sim_mat_l1) # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = sim_mat_l1.repeat(1,channels,1, 1)
        sim_mat_l1 = 2*sim_mat_l1 # (0, 1)

        # cos distance
        sim_mat_cos = x*c # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True) # (b,1,h,w)       
        sim_mat_cos = torch.tanh(sim_mat_cos) # (0, 1) (b,1,h,w)
        sim_mat_cos = sim_mat_cos.repeat(1,channels,1, 1) # (0, 1)
        
        # similarity matrix
        sim_mat = sim_mat_l1 * sim_mat_cos # (0, 1)
        
        # color embeding
        x_color = x + c*sim_mat
        
        # color features upsample
        c_up = self.conv(c)
        
        return x_color, c_up
           
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)
        
    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x
    
class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)

class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)
    
def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv1', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm1', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu1', nn.ReLU())
    ]))

def embedding_layer(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                         nn.ReLU(),
                         # nn.Conv2d(out_channels, out_channels, kernel_size=1),
                         # nn.ReLU(),
                         # nn.Conv2d(out_channels, out_channels, kernel_size=1),
                         # nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, kernel_size=1)
                         )




def conv_block1(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv1', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm1', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu1', nn.ReLU()),
    ]))

class Embedding_Net(nn.Module):
    def __init__(self, feature_size=32):
        super(Embedding_Net, self).__init__()

        self.en1_embedding = embedding_layer(feature_size, feature_size)
        self.en2_embedding = embedding_layer(feature_size*2, feature_size*2)
        self.de2_embedding = embedding_layer(feature_size*2, feature_size*2)
        self.de1_embedding = embedding_layer(feature_size, feature_size)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs, params=None):
        en1_embedding = self.en1_embedding(inputs[0])
        en2_embedding = self.en2_embedding(self.up(inputs[1]))

        de2_embedding = self.de2_embedding(self.up(inputs[2]))
        de1_embedding = self.de1_embedding(inputs[3])

        de2_embedding = self.sigmoid(de2_embedding)
        de1_embedding = self.sigmoid(de1_embedding)
        return [en1_embedding, en2_embedding, de2_embedding, de1_embedding]
        # return [de2_embedding, de1_embedding]


class MetaConvModel(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.

    out_features : int
        Number of classes (output of the model).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_channels, feature_size=32*4, d_hist=256, depth=[2, 2, 2, 2]):
        super(MetaConvModel, self).__init__()


        self.Encoder = nn.ModuleList([
            BasicConv(feature_size, feature_size, 3, 1),
            nn.Sequential(*[RB(feature_size) for _ in range(depth[0])]),
            Down_scale(feature_size),
            BasicConv(feature_size*2, feature_size*2, 3, 1),
            nn.Sequential(*[RB(feature_size*2) for _ in range(depth[1])]),
            Down_scale(feature_size*2),
            BasicConv(feature_size*4, feature_size*4, 3, 1),
            nn.Sequential(*[RB(feature_size*4) for _ in range(depth[2])]),
            Down_scale(feature_size * 4),
            BasicConv(feature_size * 8, feature_size * 8, 3, 1),
            nn.Sequential(*[RB(feature_size * 8) for _ in range(depth[2])]),
        ])
        self.conv_first = BasicConv(3, feature_size, 3, 1)

        # color hist
        self.conv_color = BasicConv(feature_size*8, 256*3, 3, 1)
        self.conv_color_b = BasicConv(feature_size*8, 256*3, 3, 1)
        self.conv_color_f = BasicConv(feature_size*8, 256*3, 3, 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, d_hist)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_hist = nn.Softmax(dim=1)
        self.pce = pce()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size


        self.en1 = conv_block(in_channels, feature_size, kernel_size=3,stride=1, padding=1, bias=True)  #256
        self.en2 = conv_block(feature_size, feature_size*2, kernel_size=3,stride=1, padding=1, bias=True) #128
        self.en3 = conv_block(feature_size*2, feature_size*4, kernel_size=3,stride=1, padding=1, bias=True)
        self.en4 = conv_block(feature_size*4, feature_size*8, kernel_size=3,stride=1, padding=1, bias=True)
        self.bottleneck = conv_block(feature_size*8, feature_size*8, kernel_size=3,stride=1, padding=1, bias=True)
        self.de4 = conv_block(feature_size*16, feature_size*8, kernel_size=3,stride=1, padding=1, bias=True)
        self.de3 = conv_block(feature_size*12, feature_size*4, kernel_size=3,stride=1, padding=1, bias=True)
        self.de2 = conv_block(feature_size*6, feature_size*2, kernel_size=3,stride=1, padding=1, bias=True) #128
        self.de1 = conv_block(feature_size*3, feature_size, kernel_size=3,stride=1, padding=1, bias=True) #256

        # self.de4 = conv_block(feature_size * 24, feature_size * 8, kernel_size=3, stride=1, padding=1, bias=True)
        # self.de3 = conv_block(feature_size * 12, feature_size * 4, kernel_size=3, stride=1, padding=1, bias=True)
        # self.de2 = conv_block(feature_size * 6, feature_size * 2, kernel_size=3, stride=1, padding=1, bias=True)  # 128
        # self.de1 = conv_block(feature_size * 3, feature_size, kernel_size=3, stride=1, padding=1, bias=True)  # 256


        self.output1 = conv_block1(feature_size, feature_size, kernel_size=3,stride=1, padding=1)
        self.output2 = conv_block1(feature_size, 1, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def sim_cl(self, x, y):

        return self.sim(x, y) / 0.5
    def encoder_c(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts

    '''
    define self.conv_color_f
    self.fc, self.softmax, self.pooling_f
    
    self.conv_color_b
    self.fc, self.softmax, self.pooling_b
    
    color_hist 
    '''
    def color_forward_b(self, x):
        x = self.conv_color_f(x)
        x = self.pooling(x)
        x = torch.reshape(x, (-1, 3, 256))  #(-1, 3, 256)
        color_hist = self.softmax_hist(self.fc(x))
        return color_hist
    
    def color_forward_f(self, x):
        x = self.conv_color_f(x)
        x = self.pooling(x)
        x = torch.reshape(x, (-1, 3, 256))  #(-1, 3, 256)
        color_hist = self.softmax_hist(self.fc(x))
        return color_hist

    def color_forward(self, x):
        x = self.conv_color(x)
        x = self.pooling(x)
        x = torch.reshape(x, (-1, 3, 256))  #(-1, 3, 256)
        color_hist = self.softmax(self.fc(x))
        return color_hist


    def forward(self, inputs, params=None):
        # features = self.features(inputs, params=self.get_subdict(params, 'features'))
    
        hist_x = self.conv_first(inputs)
        hist_x, _ = self.encoder_c(hist_x)
#         color_hist = self.color_forward(hist_x)
        color_hist_b = self.color_forward_b(hist_x)
        color_hist_f = self.color_forward_f(hist_x)
                

        en1 = self.en1(inputs, params=self.get_subdict(params, 'en1'))
        en2 = self.en2(self.pool(en1), params=self.get_subdict(params, 'en2'))
        en3 = self.en3(self.pool(en2), params=self.get_subdict(params, 'en3'))
        en4 = self.en4(self.pool(en3), params=self.get_subdict(params, 'en4'))
        bottle = self.bottleneck(self.pool(en4), params=self.get_subdict(params, 'bottleneck'))

        shortcuts = [en1,en2,en3,en4]
        shortcuts = self.pce(hist_x, shortcuts)

        de4 = self.de4(torch.cat((self.up(bottle), shortcuts[3]), dim=1), params=self.get_subdict(params, 'de4'))
        de3 = self.de3(torch.cat((self.up(de4), shortcuts[2]), dim=1), params=self.get_subdict(params, 'de3'))
        de2 = self.de2(torch.cat((self.up(de3), shortcuts[1]), dim=1), params=self.get_subdict(params, 'de2'))
        de1 = self.de1(torch.cat((self.up(de2), shortcuts[0]), dim=1), params=self.get_subdict(params, 'de1'))
        # de4 = self.de4(torch.cat((self.up(bottle), en4), dim=1), params=self.get_subdict(params, 'de4'))
        # de3 = self.de3(torch.cat((self.up(de4), en3), dim=1), params=self.get_subdict(params, 'de3'))
        # de2 = self.de2(torch.cat((self.up(de3), en2), dim=1), params=self.get_subdict(params, 'de2'))
        # de1 = self.de1(torch.cat((self.up(de2), en1), dim=1), params=self.get_subdict(params, 'de1'))

        output = self.output1(de1, params=self.get_subdict(params, 'output1'))
        output = self.output2(output, params=self.get_subdict(params, 'output2'))
        logits = self.sigmoid(output)

        pred_compact = logits
        dec2_resize = F.interpolate(de2, size=(256, 256), mode='bilinear', align_corners=False)
        embeddings = torch.cat((dec2_resize, de1), dim=1)

        return logits, output, [en1, en2, de2, de1], [color_hist_b, color_hist_f]
        # return logits, pred_compact, embeddings

        #return color_hist_b, color_hist_f
        #1/(3*256) * |color_hist_b - color_hist_b_gt|
        #color_hist_b_gt = compute_color_hist(img * (1 - gt))
        #color_hist_f_gt = compute_color_hist(img * gt)

        #np.clip(gt, 0, 1)
        #hist_gram = np.zeros((3, 256))
        #hist_gram[:,0] = 0

        #skimage color_hist
        #time.time()
        #image1, image2, color_hist(image1), save color_hist image1_hist_gram.npy
        #dataloader load image1hist_gra.py

def testmodel():
    return MetaConvModel(3, 1, feature_size=32)


# if __name__ == '__main__':
#     model = ModelMLPSinusoid()