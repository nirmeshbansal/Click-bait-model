from sklearn.feature_extraction.text import TfidfVectorizer
import _pickle as pickle
from sklearn import metrics
import pandas as pd
from sklearn import svm
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

def tokenize(text):
    lmtzr = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    l = []
    for t in tokens:
        try:
            t=float(t)
            l.append("<num>")
        except ValueError:
            l.append(lmtzr.lemmatize(t))
    return 1

fields = ['titles','clickbait']
data = pd.read_csv("Data/clickBait_Data.csv", skipinitialspace=True, usecols=fields)
print(data.head(5))