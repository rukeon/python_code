
import gensim
rt gensim
import pandas
import numpy
import glob
import sklearn
import tensorflow
import skflow
import sklearn
import time
import collections
import multiprocessing
import random
from sklearn import metrics
import sys


def read_log(path):
    log = pandas.read_table(path, header=None)
    if len(log.columns) > 4:
        log.columns = ["type", "datetime", "text", "pattern", "kkma"]
    else:
        log.columns = ["type", "datetime", "text", "pattern"]
    return(log)


def load_dataset(path, meta):
    black = [20151006, 20151007, 20151008, 20151012]
    red = [20151009, 20151010, 20151011]
    log_l = []
    for f in path:
        try:
            user = f[(f.rindex("/") + 1):f.index(".")]
#             if meta[meta.user == user].date.item() not in date_filter:
#                 continue
            day = "BLACK"
            if meta[meta.user == user].date.item() in red:
                day = "RED"
            tag = meta[meta.user == user].tag.item()
            log = read_log(f)
            hours = log.datetime.astype(str).str.slice(8, 10).astype(int)
            log["hour"] = hours
            log["tag"] = tag
            patterns = log.groupby("tag").apply(lambda x: x["pattern"].str.cat(sep=" "))
            result = patterns.to_frame(name="pattern").reset_index()
            log_l.append(result[["tag", "pattern"]])
        except (ValueError, AttributeError, pandas.parser.CParserError):
            continue
    log_df = pandas.concat(log_l)
    return(log_df)


def df2taggeddocs(df):
    docs = [((row.pattern.split()), row.tag) for index, row in df.iterrows()]    
    tagged_docs = [gensim.models.doc2vec.TaggedDocument(doc, [tag]) for doc, tag in docs]
    return(tagged_docs)


def hash_fn(x):
    random.seed(x)
#     val = random.randint(0, numpy.iinfo(numpy.uint32).max)
    val = random.randint(0, sys.maxsize)
    return(val)


def create_d2v_model(tagged_docs, n_dims, min_count, window, sample, negative, dm, n_steps):
    model = gensim.models.doc2vec.Doc2Vec(size=n_dims, min_count=min_count, window=window, sample=sample, negative=negative, dm=dm, seed=2357, workers=multiprocessing.cpu_count())
    model.build_vocab(tagged_docs)
    for epoch in range(n_steps):
        if epoch % (n_steps / 10) == 0:
            print("step: " + str(epoch))
        model.train(tagged_docs)
    return(model)


def docs2vec(d2v_model, tagged_docs):
    X = [d2v_model.infer_vector(doc.words) for doc in tagged_docs]
    y = [doc.tags[0] for doc in tagged_docs]
    return(X, y)


def test_word2vec(path, n):
    w2v_model = gensim.models.Word2Vec(size=300, min_count=1, seed=2357, workers=multiprocessing.cpu_count())
    for i in range(n):
        log = read_log(path[i])
        sentences = log.pattern.str.split(" ").tolist()
        if i < 1:
            w2v_model.build_vocab(sentences)
        else:
            w2v_model.train(sentences, total_examples=len(sentences))
    return(w2v_model)


class SingleMessageDocs(object):
    def __init__(self, path, pattern_index):
        self.path = path
        self.pattern_index = pattern_index

    def __iter__(self):
        for fp in glob.iglob(self.path):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    tokens = line.split("\t")
                    if len(tokens) < (self.pattern_index + 1):
                        continue
                    pattern = tokens[self.pattern_index]
                    if len(pattern) == 0:
                        continue
                    tag = int(tokens[0][0])
                    yield(gensim.models.doc2vec.TaggedDocument(pattern.split(), [tag]))


class DailyDocs(object):
    def __init__(self, path, pattern_index):
        self.path = path
        self.pattern_index = pattern_index

    def __iter__(self):
#         tag = "NA"
        for fp in glob.iglob(self.path):
            try:
                doc = []
                user = fp[(fp.rfind("/") + 1):]
#                 temp = fp[:fp.rfind("/")]
#                 tag = int(temp[(temp.rfind("/") + 1):])
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        tokens = line.split("\t")
                        if len(tokens) < (self.pattern_index + 1):
                            continue
                        pattern = tokens[self.pattern_index]
                        if len(pattern) == 0:
                            continue
                        words = pattern.split() + ["<EOS>"]
                        doc += words
                tag = str(hash_fn(pattern))
                yield(gensim.models.doc2vec.TaggedDocument(doc, [tag]))
            except ValueError:
                continue


class HourlyDocs(object):
    def __init__(self, path, time_index, pattern_index):
        self.path = path
        self.time_index = time_index
        self.pattern_index = pattern_index

    def __iter__(self):
        for fp in glob.iglob(self.path):
            df = pandas.read_table(fp, header=None)
            df["hour"] = df[self.time_index].astype(str).str.slice(8, 10).astype(int)
            dfgb = df.groupby("hour").apply(lambda x: (x[self.pattern_index]).str.cat(sep=" ") + "<EOS>").to_frame(name="pattern")
            patterns = dfgb.pattern.tolist()
#             temp = fp[:fp.rfind("/")]
#             tag = int(temp[(temp.rfind("/") + 1):])
#             user = fp[(fp.rfind("/") + 1):]
#             counter = 0
            for pattern in patterns:
#                 tag = user + "_" + str(counter)
#                 counter = counter + 1
                tag = str(hash_fn(pattern))
                yield(gensim.models.doc2vec.TaggedDocument(pattern.split(), [tag]))


def model_d2v(n_dims, min_count, window, sample, negative, dm, n_steps):
    start_time = time.time()
    wd = "/data/calendar/"
#     tagged_docs = DailyDocs(wd + "logs/train/*/*", 3)
    tagged_docs = HourlyDocs(wd + "logs/train/*/*", 1, 3)
    d2v_model = create_d2v_model(tagged_docs, n_dims=n_dims, min_count=min_count, window=window, sample=sample, negative=negative, dm=dm, n_steps=n_steps)
    print("> " + str(n_dims) + "\t" + str(min_count) + "\t" + str(window))
    print("> DONE: create_d2v_model [%s seconds]" % (time.time() - start_time))
#     print("> similarity = " + str(d2v_model.docvecs.most_similar(1)[0][1]))
    dm_str = "dm" if dm == 1 else "dbow"
    sample_str = str(sample)[-1]
    d2v_model.save("/data/rukeon_workspace/" + "models/cal.hourly.uniq.d" + str(n_dims) + ".c" + str(min_count) + ".w" + str(window) + ".s" + sample_str + "-" + str(negative) + ".n" + str(n_steps) + "." + dm_str + ".d2v")



model_d2v(128, 8, 4, 1e-5, 4, 0, 100)
model_d2v(200, 8, 4, 1e-5, 8, 0, 100)
model_d2v(256, 8, 4, 1e-3, 4, 0, 100)
model_d2v(300, 8, 4, 1e-3, 8, 0, 100)
