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


from customize_class import *


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = RewriteConv2d(1, 6, 5)
        self.po1 = RewriteAvgPool2d(2, 2)
        self.s1 = nn.Sigmoid()
        self.conv2 = RewriteConv2d(6, 16, 5)
        self.po2 = RewriteAvgPool2d(2, 2)
        self.s2 = nn.Sigmoid()
        self.layer1 = RewriteLinear(256, 120)
        self.s3 = nn.Sigmoid()
        self.layer2 = RewriteLinear(120, 84)
        self.s4 = nn.Sigmoid()
        self.layer3 = RewriteLinear(84, 10)

    def forward(self, input):
        batch_size, channel, h, w = input.size()
        output = self.conv1(input)
        output = self.po1(output)
        output = self.s1(output)
        output = self.conv2(output)
        output = self.po2(output)
        output = self.s2(output)
        output = output.reshape(batch_size, -1)
        output = self.layer1(output)
        output = self.s3(output)
        output = self.layer2(output)
        output = self.s4(output)
        output = self.layer3(output)
        return output

    def initialize_weight(self):

        nn.init.xavier_normal_(self.conv1.kernel)
        nn.init.xavier_normal_(self.conv2.kernel)
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.xavier_normal_(self.layer3.weight)
