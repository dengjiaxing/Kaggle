# eccoding:utf-8
import pandas as pd

# 获取数据
train = pd.read_csv('../data/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('../data/testData.tsv', delimiter='\t')
# print (train.head())
# 导入BeautifulSoup用于整洁数据
from bs4 import BeautifulSoup
import re
from  nltk.corpus import stopwords  # 导入停用词表
import xgboost


def review_to_text(review, review_stopwords):
    # 1.使用get_text函数去掉HTML标签
    raw_text = BeautifulSoup(review, 'html').get_text()
    # 2.去掉非英文字母字符
    letters = re.sub('[^a-zA-Z]', '', raw_text)
    words = letters.lower().split()
    # 3.如果remove_stopwords被激活，则进一步去掉评论中的停用词
    if review_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    return words


# 分别对原始数据做3项预处理
X_train = []
for review in train['review']:
    X_train.append(' '.join(review_to_text(review, True)))
X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review, True)))
y_train = train['sentiment']
# 导入文本特性抽取器
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 导入朴素贝叶斯模型
from  sklearn.naive_bayes import MultinomialNB
# 导入Pipeline用于方便搭建系统流程
from sklearn.pipeline import Pipeline
# 导入网格搜索用于超参数组合
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
    # 使用Pipeline搭建两组使用朴素贝叶斯分类器，区别在于分别使用CountVectorizer和TfidfVectorizer对文本特征进行抽取
    pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
    pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
    # 分别配置用于模型超参数搜索的组合
    params_count = {'count_vec__binary': [True, False], 'count_vec__ngram_range': [(1, 1), (1, 2)],
                    'mnb__alpha': [0.1, 1.0, 10.0]}
    params_tfidf = {'tfidf_vec__binary': [True, False], 'tfidf_vec__ngram_range': [(1, 1), (1, 2)],
                    'mnb__alpha': [0.1, 1.0, 10.0]}
    # 采用4折交叉验证的方法对CountVectorizer的朴素贝叶斯模型并行化超参数搜索
    gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)
    gs_count.fit(X_train, y_train)
    # 输出交叉验证中最佳的准确性得分以及超参数组合
    print(gs_count.best_score_)
    print(gs_count.best_params_)
    # 以最佳参数组合进行预测
    count_y_predict = gs_count.predict(X_test)

    # 采用4折交叉验证的方法对TfidfVectorizer的朴素贝叶斯模型并行化超参数搜索
    gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1, verbose=1)
    gs_tfidf.fit(X_train, y_train)
    # 输出交叉验证中最佳的准确性得分以及超参数组合
    print(gs_tfidf.best_score_)
    print(gs_tfidf.best_params_)
    # 以最佳参数组合进行预测
    tfidf_y_predict = gs_tfidf.predict(X_test)
    # 使用pandas对输出的数据进行格式化
    submission_count = pd.DataFrame({'id': test['id'], 'sentiment': count_y_predict})
    submission_tfidf = pd.DataFrame({'id': test['id'], 'sentiment': count_y_predict})
    # 结果输出到本地硬盘
    submission_count.to_csv('../data/submission_count.csv', index=False)
    submission_tfidf.to_csv('../data/submission_tfidf.csv', index=False)

# #从本地读入未标记数据
# unlabeled_train=train = pd.read_csv('../data/unlabeledTrainData.tsv', delimiter='\t',quoting=3)
# import nltk.data
# #使用nltk的tokenizer对影评中的英文句子进行分割
# tokenizer=nltk.data.load('tokenizers/punk/english.pickle')
# #定义函数逐条对影评进行分句
# def review_to_sentence(review,tokenizer):
#     raw_sentences=tokenizer.tokenizer(review.strip())
#     sentences=[]
#     for raw_sentence in raw_sentences:
#         if len(raw_sentence)>0:
#             sentences.append(review_to_text(raw_sentence,False))
#     return  sentences
