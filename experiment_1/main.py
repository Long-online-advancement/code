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

from dataset import *
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from visualization import *
from model import MyNet


def main():

    learning_rate = 1e-2
    epoch = 1000
    batch_size = 1

    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)

    model = MyNet()
    model.initialize_weight()

    # 读取数据，并将数据集划分为训练集，验证集，测试集。
    data_set = read_file("D:\\Course\\DeepLearning\\iris.data")
    train_data, verify_data, test_data = shuffle(data_set)

    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=detection_collate)
    verify_loader = DataLoader(verify_data, batch_size=batch_size, collate_fn=detection_collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=detection_collate)

    # 获得第一个样本
    sample = data_set[0]

    # 输出第一个样本的手动计算梯度与自动计算梯度的差
    check_grad(sample, model)

    # 初始化每一轮在验证集与测试集上的误差
    loss_train = torch.zeros(epoch)
    loss_verify = torch.zeros(epoch)

    # 初始化每20轮在测试集、验证集、预测集上的准确率
    epoch_divide20 = int(epoch/20)
    accuracy_train = torch.zeros(epoch_divide20)
    accuracy_verify = torch.zeros(epoch_divide20)
    accuracy_test = torch.zeros(epoch_divide20)

    # 开始训练epoch轮模型
    for repeat_time in range(epoch):
        for batch_data, batch_label in train_loader:

            # 训练模型
            model.manual_train(batch_data, batch_label, learning_rate)

        # 计算模型在训练集上的误差
        loss_ave = 0
        y_pred_accuracy = 0
        for data, label in train_loader:
            # 使用模型计算出结果
            y = model(data)
            y_pred = y.argmax() == torch.tensor(label)

            # 计算损失
            criterion = nn.CrossEntropyLoss()
            label = torch.tensor(label)
            loss = criterion(y, label)
            loss_ave += loss
            y_pred_accuracy += y_pred

        loss_ave = loss_ave/len(train_data)
        loss_train[repeat_time] = loss_ave
        train_accuracy = 1.0 * y_pred_accuracy / len(train_data)

        # 计算模型在验证集上的平均误差和准确率
        loss_ave = 0
        y_pred_accuracy = 0
        for data, label in verify_loader:
            # 使用模型计算出结果
            y = model(data)
            y_pred = y.argmax() == torch.tensor(label)

            # 计算损失和准确度
            criterion = nn.CrossEntropyLoss()
            label = torch.tensor(label)
            loss = criterion(y, label)
            loss_ave += loss  # 累加每个样本的误差
            y_pred_accuracy += y_pred  # 累加每个样本的预测准确度

        loss_ave = loss_ave/len(verify_data)
        loss_verify[repeat_time] = loss_ave
        verify_accuracy = 1.0 * y_pred_accuracy/len(verify_data)

        # 计算模型在预测集上的准确度
        y_pred_accuracy = 0
        for data, label in test_loader:
            # 使用模型计算出结果
            y = model(data)
            y_pred = y.argmax() == torch.tensor(label)

            # 计算准确度
            y_pred_accuracy += y_pred  # 累加每个样本的预测准确度

        test_accuracy = 1.0 * y_pred_accuracy / len(test_data)

        # 每20轮输出模型在训练集，验证集，测试集上的准确率
        if repeat_time % 20 == 0:
            num = int(repeat_time/20)
            accuracy_train[num] = train_accuracy
            accuracy_verify[num] = verify_accuracy
            accuracy_test[num] = test_accuracy

    # 最后将损失函数和准确率绘制出图像
    loss_train = loss_train.detach().numpy()
    loss_verify = loss_verify.detach().numpy()

    accuracy_train = accuracy_train.detach().numpy()
    accuracy_verify = accuracy_verify.detach().numpy()
    accuracy_test = accuracy_test.detach().numpy()

    print("最终在训练集上的损失为：%f\n在验证集上的损失为：%f\n" % (loss_train[-1], loss_verify[-1]))
    print("最终在训练集上的准确率为：%f%%\n在验证集上的准确率为：%f%%\n在测试集上的准确率为：%f%%\n"
          % (accuracy_train[-1] * 100, accuracy_verify[-1] * 100, accuracy_test[-1] * 100))
    visualize_loss(loss_train, loss_verify)
    visualize_accuracy(accuracy_train, accuracy_verify, accuracy_test)


def detection_collate(batch):
    """
    对DataLoader的传入进行重写

    """
    targets = []
    datas = []
    for sample in batch:
        datas.append(sample[0])
        targets.append(sample[1])
    datas = torch.tensor(datas)
    return datas, targets


def check_grad(sample, model):
    """
    检查手动计算的梯度与自动计算的梯度之间的差值

    """
    sample_data = torch.tensor([sample[0]], dtype=torch.float)
    sample_label = torch.tensor([sample[1]])

    # 用模型计算出结果
    y_pred = model(sample_data)

    # 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred, sample_label)

    # 将模型梯度清零
    model.zero_grad()

    # 使用手动计算梯度与自动计算梯度，比较梯度之差
    w1_grad, w2_grad, w3_grad = model.manual_backward(y_pred, sample_label)
    loss.backward()
    w3_grad_distance = model.layer3.weight.grad - w3_grad
    w2_grad_distance = model.layer2.weight.grad - w2_grad
    w1_grad_distance = model.layer1.weight.grad - w1_grad

    print('W1的梯度之差为：', w1_grad_distance)
    print('\nW2的梯度之差为：', w2_grad_distance)
    print('\nW3的梯度之差为：', w3_grad_distance)


if __name__ == "__main__":
    main()
