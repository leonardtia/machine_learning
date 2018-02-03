'''
线性回归算法 Linear regression
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

'''
线性回归算法
'''
def theta(x,y):
    '''
    利用线性回归算法求θ值
    :param x:特征值
    :param y:真实值
    :return:特征参数
    '''
    a = np.dot(x.T,y)
    b = np.power(x.T*x,-1)
    theta = np.multiply(b,a)
    return theta

def yi(x,y):
    '''
    利用线性回归算法求预测值
    :param x: 特征值
    :param y: 真实值
    :return: 预测值
    '''
    theta1 = theta(x,y)
    yi = np.multiply(theta1[0].T,x)
    return yi

def cost(x,y,theta):
    a = np.asmatrix(np.dot(x,theta.T),dtype=np.float32)
    b = y-a
    c = np.power(b,2)
    costt = np.sum(c)
    return costt
