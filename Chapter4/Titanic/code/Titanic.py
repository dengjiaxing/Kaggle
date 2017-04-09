# encoding:utf-8
import pandas as pd

# 从本地读取数据
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
# 查看数据集信息，看书否有数据集缺失
print train.info()
print test.info()
# 人工选取对预测有效的特征
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']
# 通过上面的info函数发现Age，Embarked存在缺失，所以需要补充数据
print X_train['Embarked'].value_counts()
print X_test['Embarked'].value_counts()

# 对于Embarked使用频率最高的特征值进行填充
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

# 对于Age这种类型的特征，我们习惯使用平均值或者中位数来填充缺失值，也是相对可以减少引入误差的一种填充方法
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
print X_test.info()

# 使用DictVectorizer对特征向量化
from sklearn.feature_extraction import DictVectorizer

dict_vec = DictVectorizer(sparse=False)
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
#print dict_vec.feature_names_
X_test=dict_vec.transform(X_test.to_dict(orient='record'))

#从sklearn.ensemble导入集成模型随机森林分类器
from  sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()

#从流行工具包xgboost导入XGBClassifier用于处理分类预测问题
from xgboost import XGBClassifier
xgbc=XGBClassifier()
#使用5折交叉验证进行性能评估，并获得平均分类准确性的得分
from sklearn.cross_validation import cross_val_score
print cross_val_score(rfc,X_train,y_train,cv=5).mean()
print cross_val_score(xgbc,X_train,y_train,cv=5).mean()

#使用默认配置的RandomForestClassifier进行预测操作
rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)
rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})
#将默认配置的RandomForestClassifier对测试数据的预测结果保存在文件中
rfc_submission.to_csv('../data/rfc_submission.csv',index=False)

#使用默认配置的XGBClassifier进行预测操作
xgbc.fit(X_train,y_train)
xgbc_y_predict=xgbc.predict(X_test)
xgbc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgbc_y_predict})
#将默认配置的RandomForestClassifier对测试数据的预测结果保存在文件中
xgbc_submission.to_csv('../data/xgbc_submission.csv',index=False)

#使用网格搜索方式寻找更好的超参数组合，已期待能更好的提升XGBClassifier的预测性能
from sklearn.grid_search import GridSearchCV
if __name__ == '__main__':

    params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200),
              'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
    xgbc_best = XGBClassifier()
    gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
    gs.fit(X_train, y_train)
    print  gs.best_score_
    print  gs.best_params_

    xgbc__best_y_predict = gs.predict(X_test)
    xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc__best_y_predict})
    # 将默认配置的RandomForestClassifier对测试数据的预测结果保存在文件中
    xgbc_best_submission.to_csv('../data/xgbc_best_submission.csv', index=False)