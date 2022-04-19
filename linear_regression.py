# ！/usr/bin/env python
# @Time:2022/3/22 16:02
# @Author:华阳
# @File:linear_regression.py
# @Software:PyCharm

'''
sklearn.tree.DecisionTreeClassifier()用法
criterion:不纯度的衡量指标，有基尼指数（gini）和信息熵（entropy）两种选择
max_depth:树的最大深度，超过最大深度的最大分支都会被剪掉
min_samples_leaf:一个节点被分支后每个子节点必须包含的样本个数，小于该值，分支不会发生
min_samples_split:一个节点必须包含的样本个数，小于该值，分子不会发生
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
df = pd.read_csv("boston.csv")
df = np.array(df.values,ndmin=2)
x_data = df[:,:12]
#归一化
for i in range(12):
    x_data[:,i] = (x_data[:,i]-x_data[:,i].min())/(x_data[:,i].max()-x_data[:,i].min())
y_data = df[:,12]
#将后10个作为测试集，不参加训练
test_x = x_data[-10:]
test_y = y_data[-10:]

x_d = x_data[:-10]
y_d = y_data[:-10]
#初始化参数
w = np.random.normal(0.0,1.0,(1,12))
b = 0.0
#设置训练轮次
train_epochs = 200
learing_rate = 0.001
loss_=[]
for count in range(train_epochs):
    loss=[]
    for i in range(len(x_d)):
        re = w.dot(x_d[i])+b
        err_loss = (y_d[i]-re)*(y_d[i]-re)
        err = 2*(y_d[i]-re)
        w += learing_rate*err*x_d[i] #err种re为负值，所以没有负号
        b += learing_rate*err
        #记录误差
        loss.append(abs(err_loss))
    loss_.append(sum(loss)/len(loss))
    print("第%d轮次，损失值为：%.3f"%(count+1,sum(loss)/len(loss)))
    #随机打乱训练集中的样本，防止模型出现结果和输入的位置有关的情况
    x_d,y_d = shuffle(x_d,y_d)
#打印误差的变化情况
fig1 = plt.figure(1,(5,5))
plt.plot(loss_)
plt.show()
#简单的评估，看看实际值和预测值之间的误差
sum_loss = 0
for i in range(10):
    print("true:\t{}".format(test_y[i]),end="\t")
    pre = np.dot(w,test_x[i])+b
    sum_loss += (pre-test_y[i])**2
    print("guess:\t{}".format(pre))
pre = [np.dot(w,x_d[i])+b for i in range(len(x_d)-10)]
fig2 = plt.figure(1,(5,10))
axe_train = fig2.add_subplot(211)
axe_train.set_title('distribution between true and prediction about train data')
axe_train.plot(y_d,color='red',label='true')
axe_train.plot(pre,color='blue',label='prediction')
axe_train.legend()
pre1 = [np.dot(w,test_x[i])+b for i in range(len(test_x))]
axe_test = fig2.add_subplot(212)
axe_test.set_title('distribution between true and prediction about test data')
axe_test.plot(test_y,color='red',label='true')
axe_test.plot(pre1,color='blue',label='prediction')
axe_test.legend()

print("测试集损失为%.3f"%sum_loss)
plt.show()