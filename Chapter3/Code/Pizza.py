#encoding:utf-8
#输入训练样本的特征以及目标值，分别存储在X_train和y_train中
X_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

import numpy as np
#在X轴上从0到25均匀采样100个数据点
xx=np.linspace(0,26,100)
xx=xx.reshape(xx.shape[0],1)
#以上述100个数据点作为基准，预测回归曲线
yy=regressor.predict(xx)

#对回归预测到的直线进行作图
import matplotlib.pyplot as plt
plt1,=plt.plot(xx,yy,label="Degree=1")
print "The R_squared value of Linear Regressor performing on the training data is",regressor.score(X_train,y_train)

#使用二次回归
from  sklearn.preprocessing import  PolynomialFeatures
poly2=PolynomialFeatures(degree=2)
X_train_poly2=poly2.fit_transform(X_train)
#以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，但是模型基础仍然是线性模型
regressor_poly2=LinearRegression()
regressor_poly2.fit(X_train_poly2,y_train)
xx_ploy2=poly2.transform(xx)
yy_poly2=regressor_poly2.predict(xx_ploy2)
plt2,=plt.plot(xx,yy_poly2,label="Degree=2")
print "The R_squared value of Polynominal Regressor performing on the training data is",regressor_poly2.score(X_train_poly2,y_train)

#使用四次多项式回归
from  sklearn.preprocessing import  PolynomialFeatures
poly4=PolynomialFeatures(degree=4)
X_train_poly4=poly4.fit_transform(X_train)
#以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，但是模型基础仍然是线性模型
regressor_poly4=LinearRegression()
regressor_poly4.fit(X_train_poly4,y_train)
xx_ploy4=poly4.transform(xx)
yy_poly4=regressor_poly4.predict(xx_ploy4)
plt3,=plt.plot(xx,yy_poly4,label="Degree=4")
print "The R_squared value of Polynominal4 Regressor performing on the training data is",regressor_poly4.score(X_train_poly4,y_train)

plt.scatter(X_train,y_train)
#分别对训练数据点，线性回归直线，2次多项式回归曲线作图
plt.axis([0,25,0,25])
plt.xlabel("Diameter of Pizza")
plt.ylabel("Price of Pizza")
plt.legend(handles=[plt1,plt2,plt3])
plt.show()

X_test=[[6],[8],[11],[16]]
y_test=[[8],[12],[15],[18]]
print "regressor:",regressor.score(X_test,y_test)
X_test_poly2=poly2.transform(X_test)
print "poly2:",regressor_poly2.score(X_test_poly2,y_test)
X_test_poly4=poly4.transform(X_test)
print "poly2:",regressor_poly4.score(X_test_poly4,y_test)
