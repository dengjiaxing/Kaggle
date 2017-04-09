#encoding:utf-8
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

digits_train=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)

X_train=digits_train[np.arange(64)]
y_train=digits_train[64]
X_test=digits_test[np.arange(64)]
y_test=digits_test[64]

from sklearn.cluster import KMeans

#初始化KMeans模型，并设置聚类中心点数量为10
kmeans=KMeans(n_clusters=10)
kmeans.fit(X_train,y_train)
y_pred=kmeans.predict(X_test)

from sklearn import metrics
#使用ARI进行KMeans聚类性能评估
print metrics.adjusted_rand_score(y_test,y_pred)