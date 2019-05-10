#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:08:36 2019

@author: Zane Denmon, Bin Dong, Dillon Schetley
"""

import sys
import shared_methods
import adaboost
import bagging
import voting
import numpy as np
import threading 
from sklearn import metrics

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    """Code taken from: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def fit_model(num, models, X_train, y_train): 
    '''num = 0 means adaboost
        num = 1 means bagging
        num = 2 means voting'''
    if num == 0:
        models["0"] = adaboost.OurBoostingClassifier(X_train[0], y_train[0])
    elif num == 1:
        models["1"] = bagging.OurBaggingClassifier(X_train[1], y_train[1])
    elif num == 2:
        models["2"] = voting.OurVotingClassifier(X_train[2], y_train[2])
    
    return

def main():
    if (len(sys.argv) != 5):
        print("Please provide four .csv files as arguments.")
        sys.exit()
        
    # Retreieve the filenames from the arguments
    train_filename1 = sys.argv[1]
    train_filename2 = sys.argv[2]
    train_filename3 = sys.argv[3]
    test_filename = sys.argv[4]
    
    train_files = [train_filename1, train_filename2, train_filename3]

    # Collect the training text and labels from the train files
    # X_train is a list of lists. Each list contains 1000 dataset. 
    print("Collecting Data...", end="", flush=True)
    number_of_datarow = 1500
    X_train = []
    y_train = []
    vectorizers = []
    for file in train_files:
        text, labels = shared_methods.read_csv(file)
        TFIDF_list, vectorizer = shared_methods.TFIDF_Vectorization(text[:number_of_datarow])
        vectorizers.append(vectorizer)
        X_train.append(TFIDF_list)
        y_train.append(labels[:number_of_datarow])
    print("[DONE]")
                
    # Initialize and train the models
    print("Training Ensemble...", end="", flush=True)
    models = {}

    # boosting_clf = adaboost.OurBoostingClassifier(X_train[0], y_train[0])
    # bagging_clf = bagging.OurBaggingClassifier(X_train[1], y_train[1])
    # voting_clf = voting.OurVotingClassifier(X_train[2], y_train[2])

    t0 = threading.Thread(target=fit_model, args=(0,models,X_train,y_train),) 
    t1 = threading.Thread(target=fit_model, args=(1,models,X_train,y_train),) 
    t2 = threading.Thread(target=fit_model, args=(2,models,X_train,y_train),) 

    t0.start()
    t1.start()
    t2.start()

    t0.join()
    t1.join()
    t2.join()

    boosting_clf = models["0"]
    bagging_clf = models["1"]
    voting_clf = models["2"]
    print("[DONE]")


    
    # Retrieve and softmax the training accuracies
    scores = [boosting_clf.train_accuracy, bagging_clf.train_accuracy, voting_clf.train_accuracy]
    print(scores)
    softmax_accuracy = softmax(scores)
    print(softmax_accuracy)
       
    # Collect the testing text and labels from the test file
    print("Testing Ensemble...", end="", flush=True)
    number_of_testing_samples = 500
    X_test, y_test = shared_methods.read_csv(test_filename)
    X_test_for_boosting = vectorizers[0].transform(X_test[:number_of_testing_samples]).toarray().tolist()
    X_test_for_bagging = vectorizers[1].transform(X_test[:number_of_testing_samples]).toarray().tolist()
    X_test_for_voting = vectorizers[2].transform(X_test[:number_of_testing_samples]).toarray().tolist()

    #print(len(X_test_for_bagging))

    # Predict
    boosting_pred = boosting_clf.test_model(X_test_for_boosting)
    bagging_pred = bagging_clf.test_model(X_test_for_bagging)
    voting_pred = voting_clf.test_model(X_test_for_voting)
    print("[DONE]")
    

    # Weight the predictions based on training accuracy
    print("Weighted Metrics")
    predictions = []
    for x in range(0, number_of_testing_samples):
        pos_classification = boosting_pred[x][1] * softmax_accuracy[0] + bagging_pred[x][1] * softmax_accuracy[1] + voting_pred[x][1] * softmax_accuracy[2]
        if pos_classification > 0.5:
            predictions.append('1')
        else:
            predictions.append('0')
    print("\tAccuracy: ", end="", flush=True)
    print(metrics.accuracy_score(y_test[:number_of_testing_samples], predictions))
    print("\tF1_Score: ", end="", flush=True)
    print(metrics.f1_score(y_test[:number_of_testing_samples], predictions, average='macro'))
    
    print("Unweighted Metrics")
    predictions = []
    for x in range(0, number_of_testing_samples):
        if boosting_pred[x][1] > .5:
            pos_classification_1 = 1
        else:
            pos_classification_1 = 0
        if bagging_pred[x][1] > .5:
            pos_classification_2 = 1
        else:
            pos_classification_2 = 0
        if voting_pred[x][1] > .5:
            pos_classification_3 = 1
        else:
            pos_classification_3 = 0
        if (pos_classification_1 + pos_classification_2 + pos_classification_3) > 2:
            predictions.append('1')
        else:
            predictions.append('0')
    print("\tAccuracy: ", end="", flush=True)
    print(metrics.accuracy_score(y_test[:number_of_testing_samples], predictions))
    print("\tF1_Score: ", end="", flush=True)
    print(metrics.f1_score(y_test[:number_of_testing_samples], predictions, average='macro'))

if __name__ == '__main__':
    main()