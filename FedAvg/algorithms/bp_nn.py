# -*- coding: utf-8 -*-
"""

@File ：bp_nn.py

"""

import copy
import sys

import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
sys.path.append('../')
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import chain
from models import BP  ##自定义
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   #避免jupyter崩溃

clients_wind = [f'Task1_W_Zone{i}(未打乱)'for i in range(1, 6)]
from args import args_parser  ##自定义参数


def load_data(file_name): #读取某一个文件---横向联邦学习
    df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/Wind_new/Task 1/Task1_W_Zone1_5/' + file_name + '.csv', encoding='gbk')
    columns = df.columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    MAX = np.max(df[columns[7]])
    MIN = np.min(df[columns[7]])
    df[columns[7]] = (df[columns[7]] - MIN) / (MAX - MIN) #将8列的值，标准化

    return df


def nn_seq_wind(file_name, B): #B 实则为本地批量大小
    print('data processing...')
    dataset = load_data(file_name)
    # split
    train = dataset[:int(len(dataset) * 0.6)] #前60%为训练集
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)] #中间20%为验证集
    test = dataset[int(len(dataset) * 0.8):len(dataset)] #最后20%为测试集

    def process(data): #将特征与标签分开
        data = pd.DataFrame(data)
        categorical_cols = ['jobTitle', 'gender', 'edu', 'dept']
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le  # 保存编码器
        # 定义分箱规则
        bins = [1000, 4000, 7000, 10000, 13000]
        labels = [0, 1, 2, 3]
        # 生成分类列
        data['bonus_category'] = pd.cut(data['bonus'],
                                      bins=bins,
                                      labels=labels,
                                      right=False)  # 包含左边界，不包含右边界
        # 覆盖并删除列
        data['bonus'] = data['bonus_category']  # 赋值到原列
        data.drop(columns='bonus_category', inplace=True)  # 删除分类列
        #转换成列表 https://vimsky.com/examples/usage/python-pandas-series-tolist.html
        data = data.values.tolist()
        X, Y = [], []
        for i in range(len(data)):
            train_seq = []
            train_label = []
            for c in range(0, 8):
                train_seq.append(data[i][c])
            train_label.append(data[i][8])
            X.append(train_seq)
            Y.append(train_label)

        X, Y = np.array(X), np.array(Y)

        length = int(len(X) / B) * B
        X, Y = X[:length], Y[:length]

        return X, Y

    train_x, train_y = process(train)
    val_x, val_y = process(val)
    test_x, test_y = process(test)

    return [train_x, train_y], [val_x, val_y], [test_x, test_y]


def get_val_loss(args, model, val_x, val_y): #验证集，计算损失，model即为nn
    batch_size = args.B
    batch = int(len(val_x) / batch_size) # 计算循环次数
    val_loss = []
    for i in range(batch):
        start = i * batch_size
        end = start + batch_size
        model.forward_prop(val_x[start:end], val_y[start:end])
        model.backward_prop(val_y[start:end])
    val_loss.append(np.mean(model.loss))

    return np.mean(val_loss)


def train(args, nn):
    print('training...')
    tr, val, te = nn_seq_wind(nn.file_name, args.B)
    train_x, train_y = tr[0], tr[1]
    val_x, val_y = val[0], val[1]
    nn.len = len(train_x)  # nn.len 训练集的长度
    batch_size = args.B    # 每批次大小
    epochs = args.E      # 迭代次数
    batch = int(len(train_x) / batch_size) #每一迭代，需要训练多少次
    # training
    min_epochs = 10  
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(epochs)):
        train_loss = []
        for i in range(batch):
            start = i * batch_size
            end = start + batch_size
            nn.forward_prop(train_x[start:end], train_y[start:end])
            nn.backward_prop(train_y[start:end])
        train_loss.append(np.mean(nn.loss))
        # validation
        val_loss = get_val_loss(args, nn, val_x, val_y)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(nn)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))

    return best_model


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))


def test(args, nn):
    tr, val, te = nn_seq_wind(nn.file_name, args.B)
    test_x, test_y = te[0], te[1]
    pred = []
    correct = 0  # 初始化正确预测的样本数
    total = 0    # 初始化总样本数
    batch = int(len(test_y) / args.B)
    for i in range(batch):
        start = i * args.B
        end = start + args.B
        res = nn.forward_prop(test_x[start:end], test_y[start:end])
        res = res.tolist()
        res = list(chain.from_iterable(res))
        pred.extend(res)
        # 计算当前批次的准确率
        for j in range(len(res)):
            if abs(res[j] - test_y[start + j][0]) <= 0.1:  # 误差阈值设为 0.1
                correct += 1
            total += 1

    pred = np.array(pred)
    accuracy = correct / total if total > 0 else 0  # 计算准确率
    print('mae:', mean_absolute_error(test_y.flatten(), pred), 'rmse:',
          np.sqrt(mean_squared_error(test_y.flatten(), pred)))
    print('accuracy:', accuracy)  # 输出准确率


def main():
    args = args_parser()
    for client in clients_wind:
        nn = BP(args, client)
        nn = train(args, nn)
        test(args, nn)


if __name__ == '__main__':
    main()
