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

wd = "/home/mk/data/calendar/"
meta = pd.read_table(wd + "meta/cal.dat")
train_files = glob.glob(wd + "logs/train/*.curr.log")

docVectorizer = doc2vec.Doc2Vec.load("model")
from sklearn.externals import joblib
clf = joblib.load('docSGDclf.pkl') 

def read_log_train(path):
    log = pd.read_table(path, header=None)
    log.columns = ["type", "dateTime", "text", "pattern", "kkma"]
    log["user"] = str(path[34:38])
    return(log)

trainRealFiles = glob.glob("/home/mk/data/calendar/logs/train/*.curr.log")
trainRealLogs = [read_log_train(f) for f in trainRealFiles]
trainRealLog = pd.concat(trainRealLogs)
trainRealLog = pd.merge(left = meta, right = trainRealLog, how='right', left_on='user', right_on='user')
trainRealDocs = [((row.pattern.split()), row["tag"]) for index, row in trainRealLog.iterrows()]
TaggedDocument = namedtuple('TaggedDocument', 'words tags')
taggedtrainRealDocs = [TaggedDocument(d, [c]) for d, c in trainRealDocs]

trainX = [docVectorizer.infer_vector(doc.words) for doc in taggedtrainRealDocs]

y_p_train = clf.predict_proba(trainX)
p_1_train = [p[1] for p in y_p_train]
trainRealLog['p'] = p_1_train

trainRealLog.describe()

train_1 = trainRealLog[trainRealLog.tag == 1]

p_06 = train_1[train_1.p > 0.6]

len(set(p_06.user.tolist()))

train_0 = trainRealLog[trainRealLog.tag == 0]

p_02 = train_0[train_0.p > 0.2]

len(set(p_02.user.tolist()))

train_0.describe()

