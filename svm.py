# -*- encoding: utf8 -*-
import re
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import time
import os
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
    print('Loading model ...')
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

def list_words(mes):
    words = mes.lower().split()
    return " ".join(words)

def word_clean(array, review):
    words = review.lower().split()
    meaningful_words = [w for w in words if w in array]
    return " ".join(meaningful_words)

def print_words_frequency(train_data_features):
    # Take a look at the words in the vocabulary
    vectorizer = load_model('model/vectorizer.pkl')
    if vectorizer == None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    vocab = vectorizer.get_feature_names()
    print "Words in vocabulary:", vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print "Words frequency..."
    for tag, count in zip(vocab, dist):
        print count, tag

def load_data(filename):
    res = []
    col1 = []; col2 = []

    with open(filename, 'r') as f:
        for line in f:
            if line != "\n":
                label, p, text = line.split(" ", 2)
                col1.append(label)
                col2.append(text)

        d = {"label":col1, "text": col2}
        train = pd.DataFrame(d)
    return train

def training():
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    train = load_data('general_data/train.txt')
    train_text = train["text"].values
    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    X_train = X_train.toarray()
    y_train = train["label"]
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    fit1(X_train, y_train)

def fit1(X_train,y_train):
    uni_big = SVC(kernel='rbf', C=1000)
    uni_big.fit(X_train, y_train)
    joblib.dump(uni_big, 'model/svm.pkl')

def predict_ex(mes):
    svm = load_model('model/svm.pkl')
    if svm == None:
        training()
    vectorizer = load_model('model/vectorizer.pkl')
    svm = load_model('model/svm.pkl')
    print "---------------------------"
    print "Training"
    print "---------------------------"

    # test_message = list_words(test_message) # lam thanh chu thuong
    clean_test_reviews = []
    clean_test_reviews.append(mes)
    d2 = {"message": clean_test_reviews}
    test = pd.DataFrame(d2)
    test_text = test["message"].values.astype('str')
    test_data_features = vectorizer.transform(test_text)
    test_data_features = test_data_features.toarray()
    # print test_data_features
    s = svm.predict(test_data_features)[0]
    return s

def train_main():
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    train = load_data('general_data/train.txt')
    test = load_data('general_data/test.txt')
    print test

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["label"][392], "|", train["text"][392]

    print "Data dimensions:", test.shape
    print "List features:", test.columns.values
    print "First review:", test["label"][0], "|", test["text"][0]

    train_text = train["text"].values
    test_text = test["text"].values

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    X_train = X_train.toarray()
    y_train = train["label"]
    print X_train

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label"]

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
    print "confuse matrix: \n", confusion_matrix(y_test, y_pred, labels=["EVENT", "NEVENT"])

if __name__ == '__main__':
    train_main()
    # training()

    # mes = raw_input("Custom input: ")
    # kq = predict_ex(mes)
    # print "Result: " + kq