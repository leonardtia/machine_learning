# =============================================
#    Author       : leonard tia
#    Homepage     : http://www.datajh.cn
#    E-mail       : bomb.leo@gmail.com
#
#    Description  :最小二乘法
#    Revision     :2017-12-29
#
# =============================================
import numpy as np
from scipy.optimize import leastsq
def func(p,x):
    '''
    拟合函数
    :param p:起始值
    :param x: 样本
    :return: 预测值
    '''
    theta1,theta0=p
    return theta1*x+theta0

def cost(p,x,y,s):
    '''
    损失函数
    :param p:起始值
    :param x: 样本
    :param y: 输出
    :param s: 迭代次数
    :return: 损失值
    '''
    print(s)
    return func(p,x)-y

def para(p0,Xi,Yi,s):
    '''
    最小二乘法
    :param p0:起始值
    :param xi: 输入项
    :param yi: 输出项
    :param s: 迭代次数
    :return: theta1和theta0
    '''
    Para=leastsq(cost,p0,args=(Xi,Yi,s))
    theta1,theta0=Para[0]
    return theta1,theta0

