# -*- coding: utf-8 -*-
'''
最小随机梯度下降回归 SGD regression
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
from sklearn.linear_model import SGDRegressor,SGDClassifier
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report

def SGD(x,y,test_x,test_y, loss="squared_loss",penalty="l1",alpha=0.0001,
                   tol=0.001,random_state=1,eta0=0.01,learning_rate='optimal',power_t=0.25,max_iter=1000):
    sr = SGDRegressor(loss=loss,penalty=penalty,alpha=alpha,tol=tol,random_state=random_state,eta0=eta0,
                      learning_rate=learning_rate,power_t=power_t,max_iter=max_iter)
    sr.partial_fit(x,y)
    y_pred_undersample = sr.predict(test_x)
    y_pred_undersample[(y_pred_undersample>0.5)]=1
    y_pred_undersample[(y_pred_undersample<=0.5)]=0
    i = sr.n_iter_
    Score = sr.score(test_x, test_y)
    F1 = f1_score(test_y,y_pred_undersample)
    P = precision_score(test_y, y_pred_undersample)
    R = recall_score(test_y, y_pred_undersample)
    
    tn, fp, fn, tp= confusion_matrix(test_y,y_pred_undersample).ravel()    
    return Score,F1,P,R,tn, fp, fn, tp,i

def kfold(X,y,k=10,loss="squared_loss",penalty="l1",alpha=0.0001,
                   tol=0.001,random_state=1,eta0=0.01,learning_rate='optimal',power_t=0.25,max_iter=1000):
    fold = KFold(k,shuffle=False)
    f1_result = pd.DataFrame(np.zeros((k,8)),columns=['Score','F1','P','R','TN','FP','FN','TP'])
    m=0
    for train_index, test_index in fold.split(X):
        sc,f,p,r,tn,fp,fn,tp,i =SGD(X.iloc[train_index,:],y.iloc[train_index,:],X,y,loss=loss,penalty=penalty,alpha=alpha,tol=tol,random_state=random_state,eta0=eta0,
                      learning_rate=learning_rate,power_t=power_t,max_iter=max_iter)
        f1_result.iloc[m,0]=sc
        f1_result.iloc[m,1]=f
        f1_result.iloc[m,2]=p
        f1_result.iloc[m,3]=r
        f1_result.iloc[m,4]=tn
        f1_result.iloc[m,5]=fp
        f1_result.iloc[m,6]=fn
        f1_result.iloc[m,7]=tp
        m+=1
    macro_Score = f1_result.Score.mean()
    macro_F1 = f1_result.F1.mean()
    macro_P = f1_result.P.mean()
    macro_F = f1_result.R.mean()
    micro_TN = int(f1_result.TN.mean())
    micro_FP = int(f1_result.FP.mean())
    micro_FN = int(f1_result.FN.mean())
    micro_TP = int(f1_result.TP.mean())
    return macro_Score,macro_F1,macro_P,macro_F,micro_TN,micro_FP,micro_FN,micro_TP

'''
def my_cross_validationbyC(X,y,val_list,name='',k=10,l='l2'):
    macrcos = pd.DataFrame(np.zeros((len(val_list),9)),columns=['Score','F1','P','R','TN','FP','FN','TP','val'])
    for i in range(len(val_list)):
        macro_Score,macro_F1,macro_P,macro_R,micro_TN,micro_FP,micro_FN,micro_TP = kfold(X,y,c=val_list[i],l=l,k=k)
        macrcos['Score'].iloc[i] = macro_Score
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
    plt.title('%s max F1: %s max value: %s'%(name,Max.iloc[0,1],Max.iloc[0,-1]))
    plt.ylabel('F1')
    plt.xlabel('val')
    plt.show()
    return Max.iloc[0,-1]

def my_cross_validationbyI(X,y,val_list,name='',c=1,k=10,l='l2'):
    macrcos = pd.DataFrame(np.zeros((len(val_list),9)),columns=['Score','F1','P','R','TN','FP','FN','TP','val'])
    for i in range(len(val_list)):
        macro_Score,macro_F1,macro_P,macro_R,micro_TN,micro_FP,micro_FN,micro_TP = kfold(X,y,c=c,i=val_list[i],l=l,k=k)
        macrcos['Score'].iloc[i] = macro_Score
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
'''    