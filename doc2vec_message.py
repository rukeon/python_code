import pandas as pd
import numpy as np
import gensim
from collections import namedtuple
from gensim.models import doc2vec
import glob
import random
from sklearn import datasets, cross_validation, metrics
import time
from sklearn.linear_model import SGDClassifier


def pasteLog(path):
    log = pd.read_table(path, header=None)
    user = path[(path.rindex("/") + 1):]
    tag = meta[meta.user == user].tag.item()
    log.columns = ["type", "dateTime", "text", "pattern", "kkma"]
    log.pattern = "<S> " + log.pattern + " </S>"
    log["tag"] = tag
    log.drop(log.columns[[0, 1, 4]], axis=1, inplace=True)
    return(log)

start_time = time.time()
wd = "/home/mk/data/calendar/"
meta = pd.read_table(wd + "meta/user.dat", sep = '\t', index_col = None)
trainFiles_0 = glob.glob("/data/calendar/logs/train/0/*")
trainFiles_1 = glob.glob("/data/calendar/logs/train/1/*")
trainFiles = [trainFiles_0, trainFiles_1]
trainLogs = [pasteLog(f) for files in trainFiles for f in files]
trainLog = pd.concat(trainLogs)
trainDocs = [((row.pattern.split()), row["tag"]) for index, row in trainLog.iterrows()]
taggedTrainDocs = [gensim.models.doc2vec.TaggedDocument(d, [c]) for d, c in trainDocs]
print("--- %s minutes ---" % ((time.time() - start_time) / 60))


docVectorizer = doc2vec.Doc2Vec.load("model")


from sklearn.externals import joblib
clf = joblib.load('docSGDclf.pkl') 


start_time = time.time()
trainX = [docVectorizer.infer_vector(doc.words) for doc in taggedTrainDocs]
trainY = [doc.tags[0] for doc in taggedTrainDocs]
print("--- %s minutes ---" % ((time.time() - start_time) / 60))


start_time = time.time()
y_p = clf.predict_proba(trainX)
p_1 = [p[1] for p in y_p]
trainLog['p'] = p_1
print("--- %s minutes ---" % ((time.time() - start_time) / 60))


def load_patterns(path):
    files = glob.glob(path + "*.pattern")
    patterns = []
    for f in files:
        group = "|".join(list(pd.read_table(f, header=None)[0]))
        patterns.append(group)
    return("|".join(patterns))

start_time = time.time()
neg_patterns = load_patterns(wd + "meta/patterns/retro/neg/")
trainLog["detect"] = trainLog.pattern.str.contains(neg_patterns).astype(int)
print("--- %s minutes ---" % ((time.time() - start_time) / 60))


trainLog[trainLog.detect == 1].describe()


(trainLog[abs(trainLog.p - 0.45) < 0.01])[:20]


trainLog[trainLog.p > 0.8][:20]




