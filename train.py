from sklearn.feature_extraction.text import TfidfVectorizer
import _pickle as pickle
from sklearn import metrics
import pandas as pd
from sklearn import svm
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import nltk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import base64
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_pandas import DataFrameMapper, cross_val_score

def tokenize(text):
    lmtzr = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    l = []
    for t in tokens:
        try:
            t = float(t)
            l.append("<NUM>")
        except ValueError:
            l.append(lmtzr.lemmatize(t))
    return l

#with open('wordsEn.txt') as f:
 #   voc = f.read().splitlines()

print ("\nProcessing dataset\n")
vectorizer = TfidfVectorizer(tokenizer=tokenize,
                             stop_words='english',
                             lowercase=True,
                             analyzer="word",
                             ngram_range=(1, 3))


titles = {
    "0":[],
    "1":[]
}
with open ('Data/Raw/Raw/clickbait.txt') as f:
    titles["1"] = f.read().splitlines()
with open ('Data/Raw/Raw/genuine.txt') as f:
    titles["0"] = f.read().splitlines()
train_labels = [0]*len(titles["0"]) + [1]*len(titles["1"])
train_set = titles["0"] + titles["1"]
print("Train size: " + str(len(train_labels)))
train_set = vectorizer.fit_transform(train_set, train_labels)
print  ("\nTrain matrix shape: " + str(train_set.shape))
params = {'kernel': 'rbf', 'C': 2, 'gamma': 1}
clf = svm.SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'], probability=True)
clf.fit(train_set, train_labels)
predictions = clf.predict(train_set)
print ("\nTrain set")
print ("\nAccuracy Score: " + str(metrics.accuracy_score(predictions, train_labels)))
print ("F1 Score: " + str(metrics.f1_score(predictions, train_labels)))
print ("Recall: " + str(metrics.recall_score(predictions, train_labels)))
print ("Precision: " + str(metrics.precision_score(predictions, train_labels)))




