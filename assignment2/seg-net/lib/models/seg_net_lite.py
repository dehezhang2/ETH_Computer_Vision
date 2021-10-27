from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
        up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
        pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = []
        layers_bn_down = []
        layers_pooling = []
        for i in range(self.num_down_layers):   
            layers_conv_down.append(nn.Conv2d(  
                in_channels=input_size if i==0 else down_filter_sizes[i-1],
                out_channels=down_filter_sizes[i],
                kernel_size=(kernel_sizes[0], kernel_sizes[1]),
                padding=(conv_paddings[0], conv_paddings[1])
            ))
            layers_bn_down.append(nn.BatchNorm2d(down_filter_sizes[i]))
            layers_pooling.append(nn.MaxPool2d(
                kernel_size = (pooling_kernel_sizes[0], pooling_kernel_sizes[1]), 
                stride      = (pooling_strides[0], pooling_strides[1]),
                return_indices = True
            ))

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = []
        layers_bn_up = []
        layers_unpooling = []
        for i in range(self.num_up_layers):   
            layers_unpooling.append(nn.MaxUnpool2d(
                kernel_size = (pooling_kernel_sizes[2], pooling_kernel_sizes[3]), 
                stride      = (pooling_strides[2], pooling_strides[3]),
            ))
            layers_conv_up.append(nn.Conv2d(  
                in_channels=down_filter_sizes[-1] if i==0 else up_filter_sizes[i-1],
                out_channels=up_filter_sizes[i],
                kernel_size=(kernel_sizes[2], kernel_sizes[3]),
                padding=(conv_paddings[2], conv_paddings[3])
            ))
            layers_bn_up.append(nn.BatchNorm2d(up_filter_sizes[i]))

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)
        self.relu = nn.ReLU(True)
        self.score = nn.Softmax(dim=1)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        self.conv_1_1 = nn.Conv2d(
            in_channels = up_filter_sizes[-1], 
            out_channels=11, 
            kernel_size=1,
        )

    def forward(self, x):
        indices = []
        for i in range(self.num_down_layers):
            x = self.relu(self.layers_bn_down[i](self.layers_conv_down[i](x)))
            x, index = self.layers_pooling[i](x)
            indices.append(index)

        for i in range(self.num_up_layers):
            x = self.relu(self.layers_bn_up[i](self.layers_conv_up[i](self.layers_unpooling[i](x, indices[-i-1]))))
        x = self.conv_1_1(x)
        return x

def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
