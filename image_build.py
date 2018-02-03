# =============================================
#    Author       : leonard tia
#    Homepage     : http://www.datajh.cn
#    E-mail       : bomb.leo@gmail.com
#
#    Description  :绘图
#    Revision     :2017-12-29
#
# =============================================
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(x,y,Xi,Yi,):
    '''
    散点图+折线图
    :param x: 原始坐标x
    :param y: 原始坐标y
    :param Xi: 预测坐标x
    :param Yi: 预测坐标y
    :return:
    '''
    fig,ax = plt.subplots(figsize=(6,4))
    ax.plot(Xi,Yi,c='orange',linewidth=2,label='Xi,Yi')
    ax.scatter(x,y,c='red',linewidth=3,label='x,y')
    ax.set_xlabel('x,Xi')
    ax.set_ylabel('y,Yi')
    ax.set_title('Least Squares')
    plt.legend()
    plt.show()

def plot_costs(costs,name):
    '''
    利用折线图观测损失率收敛程度
    :param costs: 损失率
    :param name: 图的名字
    :return:
    '''
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)),costs,'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper()+'Error vs. Iterations')
    plt.show()

def plot(x,y,title='image'):
    '''
    折线图
    :param x: 样本x值
    :param y: 样本y值
    :return:
    '''
    fig,ax=plt.subplots(figsize=(4,4))
    ax.plot(x,y,c='blue')
    ax.legend()
    ax.set_title(title)
    plt.show()


def scatter(x,y,title='image'):
    '''
    散点图
    :param x:样本x
    :param y: 样本y
    :return:
    '''
    fig,ax = plt.subplots(figsize=(4,4))
    ax.scatter(x,y,s=50,c='orange',marker='o')
    ax.legend()
    ax.set_title(title)
    plt.show()

def bar(x,y,title='image'):
    '''
    柱形图
    :param x:样本x
    :param y: 样本y
    :return:
    '''
    fig,ax = plt.subplots(figsize=(4,4))
    ax.bar(x,y)
    ax.legend()
    ax.set_title(title)
    plt.show()


def scatter2(x1,y1,x2,y2):
    '''
    2个样本的散点图
    :param x1:样本1的x
    :param y1:样本1的y
    :param x2:样本2的x
    :param y2:样本2的y
    :return:
    '''
    fig,ax = plt.subplots(figsize=(20,10))#画布大小长宽
    ax.scatter(x1,y1,s=50,c='b',marker='o')#散点图：x,y，点的大小，颜色，点的形状，图例说明
    ax.scatter(x2,y2,s=50,c='black',marker='x')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()