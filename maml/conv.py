import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn.modules.utils import _single, _pair, _triple
from maml.module import MetaModule

class MetaConv1d(nn.Conv1d, MetaModule):
    __doc__ = nn.Conv1d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _single(0), self.dilation, self.groups)

        return F.conv1d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)


class MetaConv2d(nn.Conv2d, MetaModule):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        # print('self.padding:', self.padding[0], self.padding[1])

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            # print('expanded_padding', expanded_padding)
            out1 = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _pair(0), self.dilation, self.groups)
            print('test classifier out by padding:', out)
            return out1

        # print('input:', input.shape)
        out = F.conv2d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)
        # print('Classifier out:', out.shape)
        return out


class MetaConv3d(nn.Conv3d, MetaModule):
    __doc__ = nn.Conv3d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv3d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _triple(0), self.dilation, self.groups)

        return F.conv3d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)


class MetaConvTranspose2d(nn.ConvTranspose2d, MetaModule):
    __doc__ = nn.ConvTranspose2d.__doc__

    def forward(self, input, params=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        output_padding = self._output_padding(input,
                                              None,
                                              self.stride,
                                              self.padding,
                                              self.kernel_size)

        return F.conv_transpose2d(input, params['weight'], bias, self.stride,
                                  self.padding, output_padding, self.groups, self.dilation)

