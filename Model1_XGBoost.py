#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing, linear_model, naive_bayes
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition, ensemble
import pandas, xgboost

#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers
from sklearn.metrics import roc_auc_score


# In[3]:


#loading the data

train_text = pd.read_json("test_instances.json", lines=True)
train_truth = pd.read_json("test_truth.json", lines=True)
test_text = pd.read_json("train_instances.json", lines=True)
test_truth = pd.read_json("train_truth.json", lines=True)


# ### Loading data and combining

# In[4]:


train = pd.merge(train_text,train_truth, on = 'id')
train.head()


# In[5]:


test = pd.merge(test_text,test_truth, on = 'id')
test.head()


# In[6]:


train = train[['postText','truthClass']]
test = test[['postText','truthClass']]
train['postText']=train['postText'].astype(str).str.replace('\[|\]|\'|\"', '')
test['postText']=test['postText'].astype(str).str.replace('\[|\]|\'|\"', '')


# In[7]:


train.head()


# In[8]:


# removing stop words is not working since 
# from nltk.corpus import stopwords
# stop = stopwords.words('english')


# In[9]:


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train['truthClass'] = encoder.fit_transform(train['truthClass'])
test['truthClass'] = encoder.fit_transform(test['truthClass'])


# In[10]:


test.head()


# In[11]:


train_x = train['postText']
train_y = train['truthClass']
test_x = test['postText']
test_y = test['truthClass']


# In[12]:


print(train_x.shape,train_y.shape, test_x.shape,test_y .shape)


# ### Feature engineering and preprocessing

# In[13]:


# create a count vectorizer object 

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train_x)

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xtest_count =  count_vect.transform(test_x)


# In[14]:


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train_x)

xtrain_tfidf =  tfidf_vect.transform(train_x)
xtest_tfidf =  tfidf_vect.transform(test_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(train_x)

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(test_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(train_x)

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_x)


# In[23]:


# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=10, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))


# In[24]:


topic_summaries


# In[25]:


print(train_x.shape,train_y.shape)


# #### Model testing

# In[29]:


def train_model(classifier, feature_vector_train, label, feature_vector_test, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return roc_auc_score(predictions, test_y)


# In[30]:


# Naive Bayes on Count Vectors
auc = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xtest_count)
print ("NB, Count Vectors: ", auc)

# Naive Bayes on Word Level TF IDF Vectors
auc = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xtest_tfidf)
print ("NB, WordLevel TF-IDF: ", auc)

# Naive Bayes on Ngram Level TF IDF Vectors
auc = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)
print ("NB, N-Gram Vectors: ", auc)

# Naive Bayes on Character Level TF IDF Vectors
auc = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)
print ("NB, CharLevel Vectors: ", auc)


# In[31]:


# Linear Classifier on Count Vectors
auc = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xtest_count)
print ("LR, Count Vectors: ", auc)

# Linear Classifier on Word Level TF IDF Vectors
auc = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xtest_tfidf)
print ("LR, WordLevel TF-IDF: ", auc)

# Linear Classifier on Ngram Level TF IDF Vectors
auc = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram)
print ("LR, N-Gram Vectors: ", auc)

# Linear Classifier on Character Level TF IDF Vectors
auc = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xtest_tfidf_ngram_chars)
print ("LR, CharLevel Vectors: ", auc)


# In[32]:


# RF on Count Vectors
auc = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xtest_count)
print ("RF, Count Vectors: ", auc)

# RF on Word Level TF IDF Vectors
auc = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xtest_tfidf)
print ("RF, WordLevel TF-IDF: ", auc)


# In[36]:


# Extereme Gradient Boosting on Count Vectors
auc = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xtest_count.tocsc())
print ("Xgb, Count Vectors: ", auc)

# Extereme Gradient Boosting on Word Level TF IDF Vectors
auc = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xtest_tfidf.tocsc())
print ("Xgb, WordLevel TF-IDF: ", auc)

# Extereme Gradient Boosting on Character Level TF IDF Vectors
auc = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xtest_tfidf_ngram_chars.tocsc())
print ("Xgb, CharLevel Vectors: ", auc)


# In[37]:


print(train_x.shape,train_y.shape)


# #### As showen above, there is big change of auc score improving after tuning, now we tune XGBoost.

# In[38]:


xgb1 = xgboost.XGBClassifier(
learning_rate =0.1,
n_estimators=1000,
max_depth=5,
min_child_weight=1,
gamma=0,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27)


# In[39]:


auc = train_model(xgb1, xtrain_tfidf_ngram_chars.tocsc(), train_y, xtest_tfidf_ngram_chars.tocsc())
print ("Xgb, CharLevel Vectors: ", auc)


# In[40]:


param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)}

gsearch1 = GridSearchCV(estimator = xgboost.XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(xtrain_tfidf_ngram_chars.tocsc(), train_y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[42]:


param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[5,6,7]
}
gsearch2 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2.fit(xtrain_tfidf_ngram_chars.tocsc(), train_y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[50]:


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=3,
 min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(xtrain_tfidf_ngram_chars.tocsc(), train_y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[53]:


xgb2 = xgboost.XGBClassifier(
learning_rate =0.1,
n_estimators=1000,
max_depth=3,
min_child_weight=5,
gamma=0.4,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27)


# In[57]:


param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
 min_child_weight=5, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch4.fit(xtrain_tfidf_ngram_chars.tocsc(), train_y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[59]:


param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

gsearch5 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
 min_child_weight=5, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch5.fit(xtrain_tfidf_ngram_chars.tocsc(), train_y)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[61]:


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
 min_child_weight=5, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch6.fit(xtrain_tfidf_ngram_chars.tocsc(), train_y)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[62]:


param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch7 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
 min_child_weight=5, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch7.fit(xtrain_tfidf_ngram_chars.tocsc(), train_y)
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_


# In[78]:


xgb3 = xgboost.XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=7,
 gamma=0.4,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.01,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)


# In[79]:


auc = train_model(xgb3, xtrain_tfidf_ngram_chars.tocsc(), train_y, xtest_tfidf_ngram_chars.tocsc())
print ("Xgb, CharLevel Vectors: ", auc)

