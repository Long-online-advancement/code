#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
# Copyright (c) Precise Pervasive Lab, University of Science and Technology of China. 2019-2020. All rights reserved.
# File name: 
# Author: Huajun Long    ID: SA20229062    Version:    Date: 2020/10/28
# Description:
# ... ...
#
# Others:   None
# History:  None

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, n, m):
        super(Linear, self).__init__()
        self.weight = torch.rand(m, n, requires_grad=True)

    def forward(self, x):
        return x.mm(self.weight.T)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        pass

    def forward(self, x):
        return 1/(1 + torch.exp(-x))


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        pass

    def forward(self, x):
        m, n = x.size()
        result = torch.zeros(m, n)
        for i in range(m):
            tmp = torch.exp(x[i, :]).sum()
            for j in range(n):
                result[i, j] = torch.exp(x[i, j]) / tmp
        return result


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        pass

    def forward(self, x, label):
        m, n = x.size()
        result = torch.zeros(m, n)
        for i in range(m):
            tmp = torch.exp(x[i, :]).sum()
            for j in range(n):
                result[i, j] = torch.exp(x[i, j])/tmp
        return result[0, label[0]]
