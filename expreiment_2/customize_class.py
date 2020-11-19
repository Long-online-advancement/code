#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
# Copyright (c) Precise Pervasive Lab, University of Science and Technology of China. 2019-2020. All rights reserved.
# File name: 
# Author: Huajun Long    ID: SA20229062    Version:    Date: 2020/11/11
# Description:
# ... ...
#
# Others:   None
# History:  None

import torch
import torch.nn as nn


class RewriteLinear(nn.Module):
    def __init__(self, n, m):
        super(RewriteLinear, self).__init__()
        self.weight = nn.Parameter(torch.rand(m, n, requires_grad=True))
        self.bais = nn.Parameter(torch.rand(m, requires_grad=True))

    def forward(self, x):

        return x.mm(self.weight.T) + self.bais


class RewriteConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super(RewriteConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True))
        self.stride = stride
        if bias:
            self.bias_value = nn.Parameter(torch.rand(self.out_channels, 1, 1, requires_grad=True))
        else:
            self.bias_value = torch.zeros(self.out_channels, 1, 1)

    def forward(self, input):

        batch_size, in_chan, in_high, in_wide = input.size()
        out_high = (in_high - self.kernel_size) // self.stride + 1
        out_wide = (in_wide - self.kernel_size) // self.stride + 1
        input_stride = input.stride()
        strides = (*input_stride[:-2], input_stride[-2] * self.stride, input_stride[-1] * self.stride, *input_stride[-2:])
        tmp = torch.as_strided(input, (batch_size, in_chan, out_high, out_wide, self.kernel_size, self.kernel_size), strides)
        output = torch.tensordot(self.kernel, tmp, dims=([1, 2, 3], [1, 4, 5])).reshape(batch_size, self.out_channels,
                                                                                        out_high, out_wide) + self.bias_value
        return output


class RewriteAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(RewriteAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = torch.ones(kernel_size, kernel_size) / (kernel_size * kernel_size)
        self.stride = stride

    def forward(self, input):

        batch_size, in_chan, in_high, in_wide = input.size()
        out_high = in_high // self.kernel_size
        out_wide = in_wide // self.kernel_size
        input_stride = input.stride()
        strides = (*input_stride[:-2], input_stride[-2] * self.stride, input_stride[-1] * self.stride, *input_stride[-2:])
        tmp = torch.as_strided(input, (batch_size, in_chan, out_high, out_wide, self.kernel_size, self.kernel_size), strides)
        output = torch.tensordot(self.kernel, tmp, dims=([0, 1], [4, 5]))

        return output

