# =============================================
#      Author   : leonard tia
#    Homepage   : http://www.datajh.cn
#    E-mail     : bomb.leo@gmail.com
#
#  Description  :
#  Revision     :
#
# =============================================
import numpy as np,pandas as pd
#import pymysql as mysql
class learn_base(object):
    def readData2csv(self,path):
        '''
        读取数据
        :param path: csv路径
        :return: dataframe数据
        '''
        data = pd.read_csv(path)
        return data

    def readData2mysql(self,table,startindex,limit,**kwargs):
        '''
        读取mysql里的数据
        :param table: 数据表
        :param startindex: 读取的其实索引
        :param limit: 读取的记录数
        :param kwargs: mysql服务器参数
        :return: dataframe数据
        '''
        #conn=mysql.Connect(kwargs)
        #cur=conn.cursor()
        #Tsql = 'select * from %s limit %s,%s' %(table,startindex,startindex+limit-1)
        #cur.execute(Tsql)
        #ret1=cur.fetchall()
        #list1=list(ret1)
        #data = pd.DataFrame(list1)
        #return data
        pass

    #将数据泛化打乱
    def shuffleData(self,data):
        '''
        通过随机混排来修改序列
        :param data: 原始数据
        :return: x：特征值，y：真实值
        '''
        np.random.shuffle(data)
        cols=data.shape[1]
        x=data[:,0:cols-1]
        y=data[:,cols-1:cols]
        return x,y

    def insert_x0(self,data):
        '''
        在第一项里增加x0=1
        :return:
        '''
        data.insert(0,'x0',1)
        return data

class result(object):
    def RR(self,y,yi):
        a = sum(np.power((y-yi),2))
        b = sum(np.power((yi-np.mean(y)),2))
        rr = 1-(a/b)
        return rr

    def right(self,y,yi):
        '''
        通过二分类预测的函数比对真实数据，计算我的正确率
        :param y: 真实值
        :param yi: 预测值
        :return: 预测正确率
        '''
        correct=[1 if ((a==1 and b==1) or (a==0 and b==0)) else 0 for (a,b) in zip(yi,y)]
        accuracy=(sum(map(int,correct))/len(correct))*100
        print('accuracy= {0}%'.format(accuracy))