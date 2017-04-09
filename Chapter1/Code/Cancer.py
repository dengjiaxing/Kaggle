#encoding:utf-8
import pandas as pd
import  matplotlib.pyplot as plt
import  numpy as np
from sklearn.linear_model import  LogisticRegression

df_train=pd.read_csv('../Data/Datasets/Breast-Cancer/breast-cancer-train.csv') #读取训练数据
df_test=pd.read_csv('../Data/Datasets/Breast-Cancer/breast-cancer-test.csv')  #读取测试数据
df_test_negative=df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']]
df_test_positive=df_test.loc[df_test['Type']==1][['Clump Thickness','Cell Size']]
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=150,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel("Clump Thickness")
plt.xlabel("Cell Size")

#画一条随机曲线
# intercept=np.random.random([1])   #随机生成一个0到1之间的数
# coef=np.random.random([2])   #随机生成量两个0到1之间的数
# lx=np.arange(0,12)
# ly=(-intercept-lx*coef[0])/coef[1]
# plt.plot(lx,ly,c='yellow')   #画一条直线
#plt.show()
lr=LogisticRegression()
#lr.fit(df_train[['Clump Thickness','Cell Size']][:10],df_train['Type'][0:10])   #使用前10个样本训练模型
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'] )          #使用前所有样本训练模型
print "测试10个训练样本的准确率：",lr.score(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
intercept=lr.intercept_
coef=lr.coef_[0,:]
lx=np.arange(0,12)
ly=(-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c='blue')
plt.show()