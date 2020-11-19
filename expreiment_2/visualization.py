#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
# Copyright (c) Precise Pervasive Lab, University of Science and Technology of China. 2019-2020. All rights reserved.
# File name: 
# Author: Huajun Long    ID: SA20229062    Version:    Date: 2020/11/14
# Description:
# ... ...
#
# Others:   None
# History:  None
import matplotlib.pyplot as plt


def visualize_loss(train, test):
    plt.figure()
    plt.plot(train, label="train loss", c="r")
    plt.plot(test, label="test loss", c="b")
    plt.title("loss change")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.savefig("..//loss.png")
    plt.show()

def visualize_accuracy(train, test):
    plt.figure()
    plt.plot(train, label="train accuracy", c="r")
    plt.plot(test, label="test accuracy", c="g")
    plt.title("accuracy change")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.savefig("..//accuracy.png")
    plt.show()


if __name__ == "__main__":
    train = [3, 4, 5, 2, 4]
    test = [2, 4, 5, 6, 3]
    visualize_loss(train, test)
    visualize_accuracy(train, test)
