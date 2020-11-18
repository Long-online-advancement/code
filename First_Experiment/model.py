#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
# Copyright (c) Precise Pervasive Lab, University of Science and Technology of China. 2019-2020. All rights reserved.
# File name: 
# Author: Huajun Long    ID: SA20229062    Version:    Date: 2020/9/27
# Description:
# ... ...
#
# Others:   None
# History:  None

import torch
from customize_class import *

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1 = Linear(4, 20)
        self.s1 = Sigmoid()
        self.layer2 = Linear(20, 30)
        self.s2 = Sigmoid()
        self.layer3 = Linear(30, 3)
        self.s3 = Softmax()

    def forward(self, x):
        self.x = x
        self.l1 = self.layer1(self.x)
        self.h1 = self.s1(self.l1)
        self.l2 = self.layer2(self.h1)
        self.h2 = self.s2(self.l2)
        self.l3 = self.layer3(self.h2)
        return self.l3

    def initialize_weight(self):
        nn.init.normal_(self.layer1.weight, 0, 0.01)
        nn.init.normal_(self.layer2.weight, 0, 0.01)
        nn.init.normal_(self.layer3.weight, 0, 0.01)

    def manual_backward(self, y_pred, target):

        column = y_pred.size()[1]
        l3_grad = torch.zeros(column)
        for i in range(column):
            if i == target[0]:
                l3_grad[i] = self.s3(self.l3)[0][i] - 1
            else:
                l3_grad[i] = self.s3(self.l3)[0][i]

        l3_grad = l3_grad.unsqueeze(-1)
        w3_grad = l3_grad.mm(self.h2)

        s2_grad = self.h2.mul(1 - self.h2)
        l2_grad = s2_grad.T.mul((self.layer3.weight.T.mm(l3_grad)))
        w2_grad = l2_grad.mm(self.h1)

        s1_grad = self.h1.mul(1 - self.h1)
        l1_grad = s1_grad.T.mul(self.layer2.weight.T.mm(l2_grad))
        w1_grad = l1_grad.mm(self.x)

        return w1_grad, w2_grad, w3_grad

    def manual_optimize(self, learning_rate, w1_grad, w2_grad, w3_grad):
        self.layer3.weight = self.layer3.weight - learning_rate * w3_grad
        self.layer2.weight = self.layer2.weight - learning_rate * w2_grad
        self.layer1.weight = self.layer1.weight - learning_rate * w1_grad

    def manual_train(self, batch_datas, batch_label, learning_rate):

        # 用模型计算出结果
        y_pred = self.forward(batch_datas)

        # 手动计算每个参数的梯度
        w1_grad, w2_grad, w3_grad = self.manual_backward(y_pred, batch_label)

        # 手动优化各个参数的梯度
        self.manual_optimize(learning_rate, w1_grad, w2_grad, w3_grad)
