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


docVectorizer = doc2vec.Doc2Vec.load("model")


from sklearn.externals import joblib
clf = joblib.load('docSGDclf.pkl') 


# working training code!!!!!!!
meta = pd.read_table("/home/mk/data/calendar/meta/user.dat", sep = '\t', index_col = None)

def read_log_train(path):
    log = pd.read_table(path, header=None)
    log.columns = ["type", "datetime", "text", "pattern", "kkma"]
    return(log)

def load_patterns(path):
    files = glob.glob(path + "*.pattern")
    patterns = []
    for f in files:
        group = "|".join(list(pd.read_table(f, header=None)[0]))
        patterns.append(group)
    return("|".join(patterns))

start_time = time.time()
wd = "/home/mk/data/calendar/"
patterns = load_patterns("/home/mk/data/rukeon_workspace/patterns/neg/")
trainCurrFiles_0 = glob.glob("/data/calendar/logs/train/0/*")
trainCurrFiles_1 = glob.glob("/data/calendar/logs/train/1/*")
trainCurrFiles = [trainCurrFiles_0, trainCurrFiles_1]
trainCurrTaggedDocs = []
X_train = []
y_train = []

for index, fs in enumerate(trainCurrFiles):
    tag = index
    for f in fs:
        log = read_log_train(f)
        user = f[(f.rindex("/") + 1):]
        y_train.append(tag)
        vectors = log.pattern.apply(lambda x: docVectorizer.infer_vector(gensim.models.doc2vec.TaggedDocument((x.split()), [tag]).words)).tolist()
        log["p"] = [p[1] for p in clf.predict_proba(vectors)]
        hours = log.datetime.astype(str).str.slice(8, 10).astype(int).tolist()
        hours = [h - 5 for h in [h + 24 if h < 5 else h for h in hours]]
        log["hour"] = hours
        state = log.iloc[0].hour
        tmp_result = [[0]] * 24
        max_p = 0
        for index, row in log.iterrows():
            if state != row.hour:
                tmp_result[state] = [max_p]
                state = row.hour
                max_p = 0
            if row.p > max_p:
                max_p = row.p
        X_train.append(tmp_result)
print("--- %s minutes ---" % ((time.time() - start_time) / 60))


# working test code!!!!!!!
start_time = time.time()
testCurrFiles_0 = glob.glob("/data/calendar/logs/test/0/*")
testCurrFiles_1 = glob.glob("/data/calendar/logs/test/1/*")
testCurrFiles = [testCurrFiles_0, testCurrFiles_1]
testCurrTaggedDocs = []
X_test = []
y_test = []

for index, fs in enumerate(testCurrFiles):
    tag = index
    for f in fs:
        log = read_log_train(f)
        user = f[(f.rindex("/") + 1):]
        y_test.append(tag)
        vectors = log.pattern.apply(lambda x: docVectorizer.infer_vector(gensim.models.doc2vec.TaggedDocument((x.split()), [tag]).words)).tolist()
        log["p"] = [p[1] for p in clf.predict_proba(vectors)]
        hours = log.datetime.astype(str).str.slice(8, 10).astype(int).tolist()
        hours = [h - 5 for h in [h + 24 if h < 5 else h for h in hours]]
        log["hour"] = hours
        state = log.iloc[0].hour
        tmp_result = [[0]] * 24
        max_p = 0
        for index, row in log.iterrows():
            if state != row.hour:
                tmp_result[state] = [max_p]
                state = row.hour
                max_p = 0
            if row.p > max_p:
                max_p = row.p
        X_test.append(tmp_result)
print("--- %s minutes ---" % ((time.time() - start_time) / 60))


X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_train = X_train.astype(float)
X_test = X_test.astype(float)


import skflow
import multiprocessing

def input_op_fn(X):
    dim = X.get_shape().as_list()
    index = 1
    X_list = skflow.ops.split_squeeze(index, dim[index], X)
    return(X_list)

def test_model(X_test, y_test, classifier, threshold):
    y_p = classifier.predict_proba(X_test)
    y_pred = [1 if p[1] > threshold else 0 for p in y_p]
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    print('Accuracy: ' + str(accuracy * 100) + "%")
    print('Precision: ' + str(precision * 100) + "%")
    print('Recall: ' + str(recall * 100) + "%")

config = skflow.addons.ConfigAddon(num_cores=multiprocessing.cpu_count(), verbose=2)
start_time = time.time()
lstm_classifier = skflow.TensorFlowRNNClassifier(rnn_size=128, batch_size=150, learning_rate= 0.0001, steps=10000, n_classes=2, cell_type='lstm', input_op_fn=input_op_fn, num_layers=1, bidirectional=False, sequence_length=None, optimizer='Adam', continue_training=True, config_addon=config)
lstm_classifier.fit(X_train, y_train)
test_model(X_test, y_test, lstm_classifier, 0.5)
print("> %s seconds" % (time.time() - start_time))


lstm_classifier.save('lstm_max_p_context')
