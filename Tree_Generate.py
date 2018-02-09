# -*- coding: utf-8 -*-
'''
决策树 Tree Generate
# =============================================
#      Author   : leonard tia
#    Homepage   : http://www.datajh.cn
#    E-mail     : bomb.leo@gmail.com
#
#  Description  :
#  Revision     :
#
# =============================================
'''
import numpy as np
import pandas as pd
import math

def Ent(D,class_name):
    '''
    样本集合D的信息熵\n
    :param D:样本集合D\n
    :param class_name 集合D的class分类(正例/反例)\n
    :return:信息熵：-sum(p*logp)\n
    '''
    D_class = D.groupby([class_name]).count().iloc[:,0]
    all_len = len(D)
    s = 0
    for i in D_class:
        pi = i/all_len
        t = -(pi*np.log2(pi))
        s +=t
    return s

def Gain(D,Ent_D,a,class_name,IsOrder=False):
    '''
    ID3(迭代二分器):属性a对于样本集D的信息增益\n
    :param D:样本集合D\n
    :param Ent_D:样本集合D的熵值\n
    :param a:属性a的名字\n
    :param class_name 集合D的class分类(正例/反例)\n
    :param IsOrder 属性a是否是连续值\n
    :return:信息增益：Ent(D) - sum(len(a在D中的class划分D_v)/len(D)*Ent(D_v))\n
    '''
    #整个样本集合的长度
    all_len = len(D)
    s = 0
    #特征a的分支结点
    if IsOrder:
        ds = order_ds(D,a)
        for d in ds:
            t = Ent(d,class_name)*len(d)/all_len
            s += t
    else:
        a_V = D.groupby([a],as_index=False).count().iloc[:,0]
        #根据:属性a的信息熵=sum(属性a的各个特征的熵值*该特征在样本中出现的概率(实例数/样本总数))
        for i in a_V:
            d = D.loc[(D[a]==i),:]
            t = Ent(d,class_name)*len(d)/all_len
            s += t
    return Ent_D - s

def IV(D_v,a,IsOrder=False):
    '''
    属性a的固有值\n
    :param D:样本集合D\n
    :param a:属性a的名字\n
    :param IsOrder 属性a是否是连续值\n
    :return:属性a的固有值\n
    '''
    iv = 0
    if IsOrder:
        ds = order_ds(D_v,a)
        for d in ds:
            v = len(d)/len(D_v)
            s = -(v * np.log2(v))
            iv +=s
    else:
        a_V = D_v.groupby([a],as_index=False).count().iloc[:,0]
        for i in a_V:
            d = D_v.loc[(D_v[a]==i),:]
            v = len(d)/len(D_v)
            s = -(v * np.log2(v))
            iv += s
    return iv

def Gain_ratio(D,Ent_D,a,class_name,IsOrder=False):
    '''
    C4.5 求属性a的增益率\n
    :param D:样本集合D\n
    :param Ent_D:样本集合D的熵值\n
    :param a:属性a的名字\n
    :param class_name 集合D的class分类(正例/反例)\n
    :param IsOrder 属性a是否是连续值\n
    :return:增益率：Gain(D,a)/IV(a)\n
    '''
    iv = IV(D,a,IsOrder)
    gain = Gain(D,Ent_D,a,class_name,IsOrder)
    return gain/iv

def Gini(D_v,class_name):
    '''
    CART 求样本D的Gini值\n
    :param D:样本集合D\n
    :param class_name 集合D的class分类(正例/反例)\n
    :return:Gini指数\n
    '''
    D_class = D_v.groupby([class_name]).count().iloc[:,0]
    all_len = len(D_v)
    g = 0
    for i in D_class:
        pi = i/all_len
        t = math.pow(pi,2)
        g +=t
    return 1-g

def Gini_index(D_v,a,class_name,IsOrder=False):
    '''
    基尼指数\n
    :param D_v:样本集合D_v\n
    :param a:属性a的名字\n
    :param class_name 集合D的class分类(正例/反例)\n
    :param IsOrder 属性a是否是连续值\n
    :return:基尼指数：sum(len(d)/len(D_v)*Gini(d))\n
    '''
    #整个样本D_v集合的长度
    all_len = len(D_v)
    g = 0
    #特征a的分支结点
    if IsOrder:
        ds = order_ds(D_v,a)
        for d in ds:
            t = Gini(d,class_name)*len(d)/all_len
            g += t
    else:
        a_V = D_v.groupby([a],as_index=False).count().iloc[:,0]
        #基尼指数
        for i in a_V:
            d = D_v.loc[(D_v[a]==i),:]
            t = Gini(d,class_name)*len(d)/all_len
            g += t
    return g

def order_ds(D_v,a):
    '''
    如果属性a是连续值，从D_v样本集中，采用二分法划分属性a为2个类别的样本集d1，d2\n
    :param D_v D_v样本集\n
    :param a 连续值属性a\n
    '''
    #是连续值，则用二分法，取小于中位数的最大值作为划分点对属性进行二分
    m = float(D_v.loc[(D_v[a]<D_v[a].median()),a].max())
    if math.isnan(m):
        m = float(D_v.loc[(D_v[a]<D_v[a].mean()),a].max())
    d1 = D_v[(D_v[a]<=m)]
    d2 = D_v[(D_v[a]>m)]
    ds = []
    ds.append(d1)
    ds.append(d2)
    return ds


def node_tree(D_v,class_name,node_list,mode='ID3',order_list=[]):
    '''
    获得第v+1分支结点（属性）\n
    :param D_v:第v分支的样本集\n
    :param class_name:第v分支结点（属性）的名字\n
    :param node_list:第v分支前面所有分支结点（属性）的名字列表\n
    :param mode:决策书的模式：ID3(信息增益)、C4.5(增益率)、CART(GINI系数)\n
    :param order_list:样本集里的连续值属性名称集合\n
    :return:分支结点信息（结点名字，信息增益值）\n
    '''
    gain_list = []
    gain_a = 0
    Ent_node = Ent(D_v,class_name)
    #求v结点之后所有属性的信息增益
    for i in D_v.columns:
        if(i in node_list):
            continue
        lis = []
        lis.append(i)
        IsOrder= i in order_list
        if mode == 'ID3':
            gain_a = Gain(D_v,Ent_node,i,class_name,IsOrder)
            lis.append(gain_a)
        elif mode == 'C4.5':
            gain_a = Gain(D_v,Ent_node,i,class_name,IsOrder)
            lis.append(gain_a)
            gain_ratio = Gain_ratio(D_v,Ent_node,i,class_name,IsOrder)
            lis.append(gain_ratio)
        elif mode == 'CART':
            gini_index = Gini_index(D_v,i,class_name,IsOrder)
            lis.append(gini_index)
            gain_a = gini_index
        gain_list.append(lis)
    if((len(gain_list) > 0) & (gain_a >0)):
        #如果信息增益>0并且还有属性可以算出增益，则证明还可以继续找下一级结点
        p_gain = pd.DataFrame(gain_list)
    else:
        #否则，证明结果已经绝对纯净，是叶子结点，返回结果的平均值
        p_gain = pd.DataFrame(np.zeros((1,2)))
        p_gain.iloc[0,0] = class_name
        p_gain.iloc[0,1] = D_v[class_name].mean()
    #返回信息增益最大的结点
    if mode == 'ID3':
        node = p_gain[(p_gain[1] == p_gain[1].max())]
    elif mode == 'C4.5':
        if p_gain.iloc[0,0] == class_name:
            #前面已判断为叶子结点
            node = p_gain[(p_gain[1] == p_gain[1].max())]
        else:
            #取信息增益大于平均值，且增益率最大的结点
            gain_mean = p_gain[1].mean()
            node = p_gain[((p_gain[1] >= gain_mean) & (p_gain[2] == p_gain[2].max()))]  
            if len(node) == 0:
                node = p_gain[(p_gain[2] == p_gain[2].max())]
    elif mode == 'CART':
        #取基尼指数最小的结点
        node = p_gain[(p_gain[1] == p_gain[1].min())]
    return node

def root_tree(D,root_Ent,class_name,mode='ID3',order_list=[]):
    '''
    获得第1分支结点（属性）\n
    :param D:样本集\n
    :param root_Ent:整个样本的信息熵\n
    :param class_name:样本集y的名字(label)\n
    :param mode:决策书的模式：ID3(信息增益)、C4.5(增益率)、CART(GINI系数)\n
    :param order_list:样本集里的连续值属性名称集合\n
    :return:分支结点信息（结点名字，信息增益值）\n
    '''
    gain_list = []
    #遍历除了label外，所有属性
    for i in D.columns:
        if(i == class_name):
            continue
        lis = []
        lis.append(i)
        IsOrder= i in order_list
        if mode == 'ID3':
            gain_a = Gain(D,root_Ent,i,class_name,IsOrder)
            lis.append(gain_a)
        elif mode == 'C4.5':
            gain_a = Gain(D,root_Ent,i,class_name,IsOrder)
            lis.append(gain_a)
            gain_ratio = Gain_ratio(D,root_Ent,i,class_name,IsOrder)
            lis.append(gain_ratio)
        elif mode == 'CART':
            gini_index = Gini_index(D,i,class_name,IsOrder)
            lis.append(gini_index)
        gain_list.append(lis)
    p_gain = pd.DataFrame(gain_list)
    #求信息增益最大的结点
    if mode == 'ID3':
        node = p_gain[(p_gain[1] == p_gain[1].max())]
    elif mode == 'C4.5':
        gain_mean = p_gain[1].mean()
        node = p_gain[((p_gain[1] >= gain_mean) & (p_gain[2] == p_gain[2].max()))]
    elif mode == 'CART':
        #取基尼指数最小的结点
        node = p_gain[(p_gain[1] == p_gain[1].min())]
    return node