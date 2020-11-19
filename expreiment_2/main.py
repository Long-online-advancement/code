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
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from model import MyNet
import time
from visualization import *


def main():

    learning_rate = 0.05
    momentum_factor = 0.7
    epoch = 20
    batch_size = 1

    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)

    model = MyNet()
    model.initialize_weight()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_factor)

    # 将数据封装
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = DataLoader(dataset1, batch_size=batch_size)
    test_loader = DataLoader(dataset2, batch_size=batch_size)

    train_loss = torch.zeros(epoch)
    test_loss = torch.zeros(epoch)
    train_accuracy = torch.zeros(epoch)
    test_accuracy = torch.zeros(epoch)

    # 开始训练模型
    for repeat_time in range(epoch):
        time_start = time.time()
        for batch_train, batch_label in train_loader:
            y = model(batch_train)
            loss = F.cross_entropy(y, batch_label)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # 计算模型在训练集上的误差
        loss_ave = 0
        y_pred_accuracy = 0
        with torch.no_grad():
            for batch_train, batch_label in train_loader:

                # 使用模型计算出结果
                y = model(batch_train)
                y_pred = sum(torch.argmax(y, dim=1) == batch_label)
                loss = F.cross_entropy(y, batch_label)
                loss_ave += loss
                y_pred_accuracy += y_pred

        loss_ave = loss_ave * batch_size/ len(dataset1)
        y_pred_accuracy = y_pred_accuracy * 1.0 / len(dataset1)
        train_loss[repeat_time] = loss_ave
        train_accuracy[repeat_time] = y_pred_accuracy

        # 计算模型在测试集上的误差
        loss_ave = 0
        y_pred_accuracy = 0

        with torch.no_grad():
            for batch_train, batch_label in test_loader:

                # 使用模型计算出结果
                y = model(batch_train)
                y_pred = sum(torch.argmax(y, dim=1) == batch_label)
                loss = F.cross_entropy(y, batch_label)
                loss_ave += loss
                y_pred_accuracy += y_pred

        loss_ave = loss_ave * batch_size / len(dataset2)
        y_pred_accuracy = y_pred_accuracy * 1.0 / len(dataset2)
        test_loss[repeat_time] = loss_ave
        test_accuracy[repeat_time] = y_pred_accuracy
        time_end = time.time()
        print("第{}轮 时间：{}".format(repeat_time+1, time_end-time_start))

    # 最后将损失函数和准确率绘制出图像
    train_loss = train_loss.detach().numpy()
    test_loss = test_loss.detach().numpy()
    train_accuracy = train_accuracy.detach().numpy()
    test_accuracy = test_accuracy.detach().numpy()

    print("最终在训练集上的损失为：%f\n在测试集上的损失为：%f\n" % (train_loss[-1], test_loss[-1]))
    print("最终在训练集上的准确率为：%f%%\n在测试集上的准确率为：%f%%\n" % (train_accuracy[-1] * 100, test_accuracy[-1] * 100))

    visualize_loss(train_loss, test_loss)
    visualize_accuracy(train_accuracy, test_accuracy)


if __name__ == "__main__":
    main()
