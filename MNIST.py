# -*- coding: utf-8 -*-
"""
编程实现K近邻算法，对MNIST数据集进行分类，打印分类的准确率

@author:郝一达
"""

import numpy as np
import operator
from sklearn.datasets import fetch_mldata
import pandas as pd

def creatdatabase(trainset = [],testset = []):
    """
    建立训练集和测试集
    (前60000个点按步长60取1000个点作为训练集
     后10000个点按步长20取500个点作为测试集)
    """
    mnist = fetch_mldata('MNIST original',transpose_data=True,data_home='python')
    X = mnist.data
    y = mnist.target
    X = np.ndarray.tolist(X)
    X = pd.DataFrame(X)
    X.insert(784,'target',y)
    data = X.as_matrix()
    for x in range(0,60000,60):
        trainset.append(data[x])
    for i in range(60000,70000,20):
        testset.append(data[i])
    '''for x in range(1000):
        for y in range(784):
            data[x][y] = float(data[x][y])
        if random.random() < 0.8:
            trainset.append(data[x])
        else:
            testset.append(data[x])'''

def distance(point1,point2):
    """计算两点之间距离"""
    length = 0
    for x in range(784):
        length += (point1[x]-point2[x])**2
    length = np.sqrt(length)
    return length

def getpoints(trainset,testpoint,k):
    """得到最近的k个点"""
    distances = []
    for x in range(len(trainset)):
        d = distance(trainset[x],testpoint)
        distances.append((trainset[x],d))
    distances.sort(key = operator.itemgetter(1))
    points = []
    for i in range(k):
        points.append(distances[i][0])
    return points

def getmarks(points):
    """得到最多的分类"""
    T = {}
    for x in range(len(points)):
        mark = points[x][-1]
        if mark in T:
            T[mark] += 1
        else:
            T[mark] = 1
    marks = sorted(T.items(),key=operator.itemgetter(1),reverse=True)
    return marks[0][0]

def getaccuracy(testset,figure):
    """计算准确率"""
    num = 0
    for i in range(len(testset)):
        if testset[i][-1] == figure[i]:
            num += 1
    rate = (num/float(len(testset)))*100
    return rate

if __name__ == '__main__':
    trainset = []
    testset = []
    creatdatabase(trainset,testset)
    k = 3     #k取3
    figure = []
    num = 0
    for x in range(len(testset)):
        points = getpoints(trainset,testset[x],k)
        marks = getmarks(points)
        figure.append(marks)
        num += 1
        print(num)
    accuracy = getaccuracy(testset,figure)
    print('Accuracy:' + str(accuracy) + '%')