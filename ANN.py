# 利用keras构建MLP进行二分类：印第安人糖尿病预测

# 2.1 引入相关模块
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# 2.2 准备数据
df = pd.read_csv('pima_data.csv', header=None)
data = df.values
X = data[:, :-1]
y = data[:, -1]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
# 2.3 构建网络模型：定义输入层、隐含层、输出层神经元个数，采用的激活函数
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 2.4 编译模型：确定损失函数，优化器，以及评估指标
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 2.5 训练模型：确定迭代的次数，批尺寸，是否显示训练过程
model.fit(X_train, y_train, epochs=100, batch_size=20, verbose=True)
# 2.6 评估模型
score = model.evaluate(X_test,y_test,verbose=False)
print("准确率为:{:.2f}%".format(score[1]*100))