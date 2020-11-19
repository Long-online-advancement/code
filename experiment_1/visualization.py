#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
# Copyright (c) Precise Pervasive Lab, University of Science and Technology of China. 2019-2020. All rights reserved.
# File name: 
# Author: Huajun Long    ID: SA20229062    Version:    Date: 2020/10/26
# Description:
# ... ...
#
# Others:   None
# History:  None

import matplotlib.pyplot as plt


def visualize_loss(train, verify):
    plt.figure()
    plt.plot(train, label="train loss", c="r")
    plt.plot(verify, label="verify loss", c="b")
    plt.title("loss change")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.show()


def visualize_accuracy(train, verify, test):
    plt.figure()
    plt.plot(train, label="train accuracy", c="r")
    plt.plot(verify, label="verify accuracy", c="b")
    plt.plot(test, label="test accuracy", c="g")
    plt.title("accuracy change")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.show()


if __name__ == "__main__":
    train = [3, 4, 5, 2, 4]
    verify = [3, 5, 4, 3, 7]
    test = [2, 4, 5, 6, 3]
    visualize_loss(train, verify)
    visualize_accuracy(train, verify, test)
