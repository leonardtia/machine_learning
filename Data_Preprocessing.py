#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:12:00 2018

@author: leonard_tia
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import random
from imblearn.over_sampling import SMOTE

def import_dataset(path,dateindex=''):
    '''
    装载数据
    :param path: csv路径
    :return: dataframe数据
    '''
    if dateindex == '':
        dataset = pd.read_csv(path)
    else:
        dataset = pd.read_csv(path,parse_dates=[dateindex], dayfirst=True, index_col=dateindex)
    return dataset


def get_K_train(dataset,k):
    '''
    K维交叉验证法的训练集创建
    '''
    temp_len = len(dataset)
    r_len = int(temp_len/k)
    list_train = []
    for i in range(k):
        if temp_len - r_len * (i+1) < r_len:
            r_len = temp_len - r_len * i
        random_temp = np.random.choice(dataset.index,r_len,replace = False)
        k_train = dataset.loc[random_temp]
        dataset.drop(random_temp,inplace=True)
        list_train.append(k_train)
    return list_train

def get_diy_train(dataset):
    '''
    自助采样法，必然有0.368个数据是包外估计，在原始训练集里，但不在自助采样训练集里
    生成的采样集长度与样本集长度相同
    '''
    len_d = len(dataset)
    list_diy = []
    for i in range(len_d):
        random_t = np.random.choice(dataset.index,1,replace = False)
        list_diy.append(dataset.loc[random_t])
    diy_train = pd.concat(list_diy)
    return diy_train
def get_diy_trainbyxy(x,y):
    '''
    自助采样法，必然有0.368个数据是包外估计，在原始训练集里，但不在自助采样训练集里
    生成的采样集长度与样本集长度相同
    '''
    len_d = len(x)
    list_x = []
    list_y = []
    for i in range(len_d):
        random_t = np.random.choice(x.index,1,replace = False)
        list_x.append(x.loc[random_t])
        list_y.append(y.loc[random_t])
    diy_x_train = pd.concat(list_x)
    diy_y_train = pd.concat(list_y)
    diy_x_test = x.drop(diy_x_train.index)
    diy_y_test = y.drop(diy_y_train.index)
    return diy_x_train,diy_x_test,diy_y_train,diy_y_test

def get_over_sample_X_y(x,y,rs=0):
    '''
    过采样法，以多数class的len作为少数class的len,
    用最近邻的方式生产少数class，达到样本均衡的目的
    '''
    oversampler = SMOTE(random_state=rs)
    over_x,over_y = oversampler.fit_sample(x,y)
    return over_x,over_y

def get_X_y(dataset):
    '''
    获得自变量X矩阵和因变量y向量，默认数据集最后一列是因变量y
    :param dataset:数据集 
    :return: 自变量X矩阵，因变量y向量
    '''
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    return X,y

def missing_data(dataset,MissType,axis=0):
    '''
    消除数据集中的缺失值
    :param dataset:要处理的数据集
    :param MissType:要处理缺失数据的方法：
        ‘mean’：平均值，默认是列的平均值，
        ‘median’：中位数，
        ‘most_frequent’：众数，
        ‘dropna’：凡是带有Nan的行，全部丢弃,
        'dropall'：整行都待用NaN的行，丢弃
    :param axis:轴：0代表列，1代表行：
    :return: 返回处理好的数据集
    '''
    if MissType == 'dropna':
        return dataset.dropna()
    elif MissType == 'dropall':
        return dataset.dropna(how='all')
    else :
        imputer = Imputer(missing_values='NaN',strategy=MissType,axis=axis)
        imputer = imputer.fit(dataset)
        return imputer.transform(dataset)
    
def categorical_data(dataset,is_order,index=0):
    '''
    数据分类处理
    :param dataset:要进行分类处理的数据集
    :param is_order:
        True：待处理的数据集是有序关系（1，2，3）
        False：无序关系，k维数组，（1，0，1）（0，0，1）
    :param index:无序分类时，要处理的自变量index
    :return: 返回处理好的数据集
    '''
    if is_order:
        labelencoder_data = LabelEncoder()
        #将dataset转化成lael编码
        return labelencoder_data.fit_transform(dataset).toarray()
    else:
        #对dataset进行转化
        onehotencoder = OneHotEncoder(categorical_features=[index])
        #拟合并转化dataset，并将其转成向量
        return onehotencoder.fit_transform(dataset).toarray()
    
def get_TrainingSet_and_Test_set(X,y,Test_Percent=0.2):
    '''
    根据数据集划分训练集和测试集
    :param X:要进行划分的数据集自变量X
    :param y:要进行划分的数据集因变量y
    :param Test_Percent:测试集占数据集百分比
    :return: 训练集和测试集的自变量X，因变量y
    '''
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=Test_Percent)
    return X_train,X_test,y_train,y_test

def get_TrainingSet_and_Test_set2(dataset, Test_Percent=0.3):
    '''
    根据数据集划分训练集和测试集
    :param dataset:要进行划分的数据集
    :param Test_Percent:测试集占数据集百分比
    :return: 训练集和测试集
    '''
    trainSize = int(len(dataset) * Test_Percent)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def Feature_Scaling(dataset,scaling_type='Stand'):
    '''
    数据集归一化
    :param dataset:要进行归一化的数据集
    :param scaling_type:归一化的类型：
        Stand：标准化,让值符合标准正态分布
        Normal：普通化，让值符合【0，1】区间
    :return: 返回归一化后的数据集
    '''
    if scaling_type == 'Stand':
        sc_data = StandardScaler()
        return sc_data.fit_transform(dataset)
    else:
        mm_data = MinMaxScaler()
        return mm_data.fit_transform(dataset)
    

def shuffleData(dataset):
    '''
    通过随机混排来修改矩阵
    :param dataset: 原始数据
    :return: 打乱顺序后的矩阵
    '''
    np.random.shuffle(dataset)
    return dataset

def insert_x0(dataset):
    '''
    在第一项里增加x0=1
    :param dataset: 原始数据
    :return:
    '''
    x0 = np.ones((dataset.shape[0],1))
    return np.hstack((x0,dataset))