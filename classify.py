#!/usr/bin/env python
# coding: utf-8

import time,json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC,LinearSVC
from random import randint
import numpy as np
import logging
from sklearn.model_selection import GridSearchCV

#get_ipython().run_line_magic('run', 'utils.ipynb')
from utils import get_class_labels

logger = logging.getLogger()
logger.setLevel("INFO")

def subgraph2vec_tokenizer (s):
    '''
    Tokenize the string from subgraph2vec sentence (i.e. <target> <context1> <context2> ...). Just target is to be used
    and context strings to be ignored.
    :param s: context of graph2vec file.
    :return: List of targets from graph2vec file.
    '''
    return [line.split(' ')[0] for line in s.split('\n')]


def linear_svm_classify (X_train, Y_train, X_test, Y_test):

    params = {'C':[0.01,0.1,1,10,100,1000]}
    classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='f1',verbose=1)
    classifier.fit(X_train,Y_train)
    logging.info('best classifier model\'s hyperparamters', classifier.best_params_)

    Y_pred = classifier.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    logging.info('Linear SVM accuracy: {}'.format(acc))
    logging.info(classification_report(Y_test, Y_pred))
    
    return acc


def perform_classification (Xtrain, Ytrain, Xtest, Ytest, embedding_fname, embedding_fname_test):
    '''
    Perform classification from
    :param corpus_dir: folder containing subgraph2vec sentence files
    :param extn: extension of subgraph2vec sentence files
    :param embedding_fname: file containing subgraph vectors in word2vec format (refer Mikolov et al (2013) code)
    :param class_labels_fname: files containing labels of each graph
    :return: None
    '''
    with open(embedding_fname,'r') as fh:
        graph_embedding_dict = json.load(fh)
    X = np.array([graph_embedding_dict[fname] for fname in Xtrain])

    with open(embedding_fname_test, 'r') as fh:
        graph_embedding_dict_test = json.load(fh)
    X_test = np.array([graph_embedding_dict_test[fname] for fname in Xtest])
    
    a = linear_svm_classify(X, Ytrain, X_test, Ytest)
    return a

def perform_classification_final_new(Xtrain, Ytrain, Xtest, Ytest, embedding_fname):
    
    with open(embedding_fname, 'r') as fh:
        graph_embedding_dict = json.load(fh)
    
    seed = randint(0, 1000)
    x_train = np.array([graph_embedding_dict[fname] for fname in Xtrain])
    x_test  = np.array([graph_embedding_dict[fname] for fname in Xtest])
    
    #X_train, X_test, Y_train, Y_test = train_test_split(Xnew, Y, test_size=0.1, random_state=seed)
    
    params = {'C':[0.01,0.1,1,10,100,1000]}
    classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='f1',verbose=1)
    classifier.fit(x_train, Ytrain)
    logging.info('best classifier model\'s hyperparamters', classifier.best_params_)

    Y_pred = classifier.predict(x_test)

    acc = accuracy_score(Ytest, Y_pred)
    logging.info('Linear SVM accuracy: {}'.format(acc))
    logging.info(classification_report(Ytest, Y_pred))
    
    return acc

def perform_classification_final(X, Y, embedding_fname):
    
    with open(embedding_fname, 'r') as fh:
        graph_embedding_dict = json.load(fh)
    
    seed = randint(0, 1000)
    Xnew = np.array([graph_embedding_dict[fname] for fname in X])
    X_train, X_test, Y_train, Y_test = train_test_split(Xnew, Y, test_size=0.1, random_state=seed)
    
    params = {'C':[0.01,0.1,1,10,100,1000]}
    classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='f1',verbose=1)
    classifier.fit(X_train, Y_train)
    logging.info('best classifier model\'s hyperparamters', classifier.best_params_)

    Y_pred = classifier.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    logging.info('Linear SVM accuracy: {}'.format(acc))
    logging.info(classification_report(Y_test, Y_pred))
    
    return acc

