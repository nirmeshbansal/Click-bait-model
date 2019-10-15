from sklearn.feature_extraction.text import TfidfVectorizer
import _pickle as pickle
from sklearn import metrics
#import model_selection
#import dataset_builder
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
