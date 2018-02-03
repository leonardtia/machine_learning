# -*- coding: utf-8 -*-
'''
逻辑回归算法 Logical regression
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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report
def logistic(x,y,test_x,test_y, C=1,I=0.5,L='l2',isProba=True):
    '''
    逻辑回归
    :param x:训练集feature
    :param y:训练集label
    :param test_x:测试集feature
    :param test_y:测试集leabel
    :param c:正则惩罚系数
    :param i:概率阈值
    :param l:正则惩罚模式
    :param isProba:用测试集验证结果时，是直接预测概率值还是label值
    :return:F1（2*P*R/(P+R)）,P(查准率：TP/TP+FP)，R（查全率:TP/TP+FN），
    TN（真反例），FP（假正例）,FN（假反例）,TP(真正例)
    '''
    lr = LogisticRegression(penalty=L,dual=False,C=C)
    lr.fit(x,y)
    if isProba:
        y_pred_undersample_proba = lr.predict_proba(test_x)#预测百分比
        y_pred_undersample = y_pred_undersample_proba[:,1] > I#将值=1的百分比与设置的阈值进行比较
    else:
        y_pred_undersample = lr.predict(test_x)#直接预测结果
    F1 = f1_score(test_y,y_pred_undersample)
    P = precision_score(test_y, y_pred_undersample)
    R = recall_score(test_y, y_pred_undersample)
    
    tn, fp, fn, tp= confusion_matrix(test_y,y_pred_undersample).ravel()    
    return F1,P,R,tn, fp, fn, tp


def kfold(X,y,c=1,i=0.5,k=10,l='l2'):
    '''
    交叉验证，找出最合适的参数
    :param X: DataFrame,特征集
    :param y: DataFrame,标记集
    :param c: float,正则惩罚系数
    :param i: float,概率阈值
    :param k: k,k维
    :param l: 正则惩罚模式
    :return: macro_F1,macro_P,macro_F,micro_TN,micro_FP,micro_FN,micro_TP
    '''
    fold = KFold(k,shuffle=False)
    f1_result = pd.DataFrame(np.zeros((k,7)),columns=['F1','P','R','TN','FP','FN','TP'])
    m=0
    for train_index, test_index in fold.split(X):
        f,p,r,tn,fp,fn,tp =logistic(X.iloc[train_index,:],y.iloc[train_index,:],X,y,C=c,I=i,L=l)
        f1_result.iloc[m,0]=f
        f1_result.iloc[m,1]=p
        f1_result.iloc[m,2]=r
        f1_result.iloc[m,3]=tn
        f1_result.iloc[m,4]=fp
        f1_result.iloc[m,5]=fn
        f1_result.iloc[m,6]=tp
        m+=1
    macro_F1 = f1_result.iloc[:,0].mean()
    macro_P = f1_result.iloc[:,1].mean()
    macro_F = f1_result.iloc[:,2].mean()
    micro_TN = int(f1_result.TN.mean())
    micro_FP = int(f1_result.FP.mean())
    micro_FN = int(f1_result.FN.mean())
    micro_TP = int(f1_result.TP.mean())
    return macro_F1,macro_P,macro_F,micro_TN,micro_FP,micro_FN,micro_TP

def my_cross_validationbyC(X,y,val_list,name='',k=10,l='l2'):
    '''
    交叉验证视觉展示，并找出最优正则化惩罚参数
    :param X: DataFrame,特征集
    :param y: DataFrame,标记集
    :param val_list: list,正则惩罚系数表
    :param name: 采样名字(english)
    :param k: k,k维
    :param l: 正则惩罚模式
    :return: Max（F1值）时对应的val
    '''
    macrcos = pd.DataFrame(np.zeros((len(val_list),8)),columns=['F1','P','R','TN','FP','FN','TP','val'])
    for i in range(len(val_list)):
        macro_F1,macro_P,macro_R,micro_TN,micro_FP,micro_FN,micro_TP = kfold(X,y,c=val_list[i],l=l,k=k)
        macrcos['F1'].iloc[i] = macro_F1
        macrcos['P'].iloc[i] = macro_P
        macrcos['R'].iloc[i] = macro_R
        macrcos['TN'].iloc[i] = micro_TN
        macrcos['FP'].iloc[i] = micro_FP
        macrcos['FN'].iloc[i] = micro_FN
        macrcos['TP'].iloc[i] = micro_TP
        macrcos['val'].iloc[i] = val_list[i]
    Max = macrcos[(macrcos.F1==macrcos.F1.max())]
    plt.figure(figsize=(8,6))
    plt.plot(val_list,macrcos.F1,'r--',label='Macro F1')
    plt.legend(loc='best')
    plt.title('%s max F1: %s max value: %s'%(name,Max.iloc[0,0],Max.iloc[0,-1]))
    plt.ylabel('F1')
    plt.xlabel('C')
    plt.show()
    return Max.iloc[0,-1]

def my_cross_validationbyI(X,y,val_list,name='',c=1,k=10,l='l2'):
    '''
    交叉验证视觉展示，并找出概率阈值最优参数
    :param X: DataFrame,特征集
    :param y: DataFrame,标记集
    :param val_list: list,正则惩罚系数表
    :param name: 采样名字(english)
    :param c: 正则化惩罚参数
    :param k: k维
    :param l: 正则惩罚模式
    :return: Max（F1值）时对应的val
    '''
    macrcos = pd.DataFrame(np.zeros((len(val_list),8)),columns=['F1','P','R','TN','FP','FN','TP','val'])
    for i in range(len(val_list)):
        macro_F1,macro_P,macro_R,micro_TN,micro_FP,micro_FN,micro_TP = kfold(X,y,c=c,i=val_list[i],l=l,k=k)
        macrcos['F1'].iloc[i] = macro_F1
        macrcos['P'].iloc[i] = macro_P
        macrcos['R'].iloc[i] = macro_R
        macrcos['TN'].iloc[i] = micro_TN
        macrcos['FP'].iloc[i] = micro_FP
        macrcos['FN'].iloc[i] = micro_FN
        macrcos['TP'].iloc[i] = micro_TP
        macrcos['val'].iloc[i] = val_list[i]
    Max = macrcos[(macrcos.F1==macrcos.F1.max())]
    plt.figure(figsize=(8,6))
    plt.plot(val_list,macrcos.F1,'r--',label='Macro F1')
    plt.legend(loc='best')
    plt.title('%s max F1: %s max value: %s'%(name,Max.iloc[0,0],Max.iloc[0,-1]))
    plt.ylabel('F1')
    plt.xlabel('I')
    plt.show()
    return Max.iloc[0,-1]
