# -*- encoding: utf8 -*-
import re

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import time
import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"

def load_model(model):
    print('loading model ...')
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None

def clean_str_vn(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[~`@#$%^&*-+]", " ", string)
    def sharp(str):
        b = re.sub('\s[A-Za-z]\s\.', ' .', ' '+str)
        while (b.find('. . ')>=0): b = re.sub(r'\.\s\.\s', '. ', b)
        b = re.sub(r'\s\.\s', ' # ', b)
        return b
    string = sharp(string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_str_en(string):
    string = re.sub(r"[~`@#$%^&*-+–]", " ", string)
    return string

def review_to_words(review, filename):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    meaningful_words = [w for w in words if not w in array]
    return " ".join(meaningful_words)


def word_clean(array, review):
    words = review.lower().split()
    meaningful_words = [w for w in words if w in array]
    return " ".join(meaningful_words)


def ngrams(input, n):
  input = input.split(' ')
  output = []
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output # output dang ['a b','b c','c d']

def ngrams2(input, n):
  input = input.split(' ')
  output = {}
  for i in range(len(input)-n+1):
    g = ' '.join(input[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
  return output # output la tu dien cac n-gram va tan suat cua no {'a b': 1, 'b a': 1, 'a a': 3}

def ngrams_array(arr,n):
    output = {}
    for x in arr:
        d = ngrams2(x, n)  # moi d la 1 tu dien
        for x in d:
            count = d.get(x)
            output.setdefault(x, 0)
            output[x] += count
    return output



def load_data(filename):
    col1 = []; col2 = []

    with open(filename, 'r') as f:
        for line in f:
            if line != "\n":
                label1, p, question = line.split(" ", 2)
                question = review_to_words(question,'dict_data/question_stopwords.txt')
                question = clean_str_en(question)
                # question = review_add_pos(question,'datavn/question_stopwords.txt')
                col1.append(label1)
                col2.append(question)

        col3 = []
        for q in col2:
            r1 = []; r2 = []
            text = nltk.word_tokenize(q)    # q la 1 cau, text: list da tach tu trong cau
            a = nltk.pos_tag(text)  # a: [('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO')]
            s1 = []  # s1 la list chua cac nhan

            # I_N go_V to_NN school_N
            for tup in a[:len(a)]:
                k = tup[0]
                # k = unicode(k, errors='replace')
                # k.decode(encoding='utf-8',errors='strict')
                t = k + "_" + tup[1] # t = They_PRP
                r1.append(t)
                s1.append(tup[1])

                # s1 = [N,V,NN,N]
            s2 = ' '.join(i for i in s1)  # Nhan tu loai cua cau dang str. vd: "N V E N"
            q2 = [' '.join(x) for x in ngrams(q, 2)]  # q2: mang cac phan tu 2-grams la word
            s22 = [' '.join(x) for x in ngrams(s2, 2)]  # s22: mang cac phan tu 2-grams la tag
            q3 = (' '.join(x.replace(' ', '_') for x in q2)).split()
            s3 = (' '.join(x.replace(' ', '_') for x in s22)).split()
            for i1, i2 in zip(q3, s3):
                t2 = i1 + "_" + i2
                r2.append(t2)

            y = r1 + r2
            z = ' '.join(y)
            z = unicode(z, errors='replace')
            col3.append(z)

        d = {"label1":col1, "question": col3}
        train = pd.DataFrame(d)
    return train

if __name__ == '__main__':
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    train = load_data('general_data/train.txt')
    test = load_data('general_data/test.txt')

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["label1"][392], "|", train["question"][392]

    print "Data dimensions:", test.shape
    print "List features:", test.columns.values
    print "First review:", test["label1"][0], "|", test["question"][0]

    train_text = train["question"].values
    test_text = test["question"].values
    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    X_train = X_train.toarray()
    y_train = train["label1"]


    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]

    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    t0 = time.time()
    # iterate over classifiers

    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print y_pred

    print " accuracy: %0.3f" % accuracy_score(y_test, y_pred)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    print "confuse matrix: \n", confusion_matrix(y_test, y_pred, labels=["EVENT","NEVENT"])
