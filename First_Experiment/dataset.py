#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
# Copyright (c) Precise Pervasive Lab, University of Science and Technology of China. 2019-2020. All rights reserved.
# File name: 
# Author: Huajun Long    ID: SA20229062    Version:    Date: 2020/10/20
# Description:
# ... ...
#
# Others:   None
# History:  None

from torch.utils.data import Dataset
import random


class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        values, label = self.data[index]
        return values, label

    def __len__(self):
        return len(self.data)


def read_file(txt_path):
    fh = open(txt_path, 'r')
    data = []
    for line in fh:
        line = line.split(",")
        label = line[-1].split()
        onehot_label = str_to_int(label)
        values = list(map(float, line[0: -1]))
        data.append((values, onehot_label))
    data.pop()
    return data


def shuffle(data):
    train_proportion = 0.8
    verify_proportion = 0.1
    test_proportion = 0.1

    random.shuffle(data)  # 将date数据集打乱
    length = len(data)

    verify = data[0: int(verify_proportion * length)]
    test = data[int(verify_proportion * length): int(verify_proportion * length) + int(test_proportion * length)]
    train = data[int(verify_proportion * length) + int(test_proportion * length):]

    return train, verify, test


def str_to_int(label):
    if label == ["Iris-setosa"]:
        return 0
    elif label == ["Iris-versicolor"]:
        return 1
    elif label == ["Iris-virginica"]:
        return 2


if __name__ == "__main__":
    dataset = MyDataSet("F:\\我的文档\\研究生\\研一课程\\深度学习\\code\\iris.data")
