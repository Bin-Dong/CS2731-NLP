#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:08:36 2019

@author: Zane Denmon, Bin Dong, Dillon Schetley
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics

class OurVotingClassifier():
    def __init__(self, X_train, y_train):
        self.model = None
        self.train_accuracy = -1
        #self.test_accuracy = -1
        self.train_model(X_train, y_train)
        
    def train_model(self, X_train, y_train):
        # Create classifiers for voting classifier object
        clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                  random_state=1)
        clf2 = RandomForestClassifier(n_estimators=140)
        clf3 = GaussianNB()
        
        # Create voting classifier object
        vc = VotingClassifier(estimators=[
                ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
        
        # Train boosting classifer
        training_data = X_train[:2500]
        training_data_labels = y_train[:2500]
        validation = X_train[2500:]
        validation_labels = y_train[2500:]
        self.model = vc.fit(training_data, training_data_labels)
        self.train_accuracy = metrics.accuracy_score(validation_labels, self.model.predict(validation))
      
    def test_model(self, X_test):
        #Predict the response for test dataset
        y_pred = self.model.predict_proba(X_test)
        return y_pred
        #self.test_accuracy = metrics.accuracy_score(y_test, y_pred)
        
## Load data
#text_list, label_list = shared_methods.read_csv("yelp_reviews.csv")
#text_list, label_list = shuffle(text_list, label_list)
#
#text_list_reduce = text_list[:4000]
#label_list_reduce = label_list[:4000]
##print(label_list_reduce[:30])
#
#TFIDF_list, vectorizer = shared_methods.TFIDF_Vectorization(text_list_reduce)
#X_sparse = coo_matrix(TFIDF_list)
#
## Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(TFIDF_list, label_list_reduce, test_size=0.8) # 70% training and 30% test
#
#clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
#                          random_state=1)
#clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
#clf3 = GaussianNB()
#
#eclf1 = VotingClassifier(estimators=[
#        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
#eclf1 = eclf1.fit(X_train, y_train)
#y_pred = eclf1.predict(X_test)
#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#eclf2 = VotingClassifier(estimators=[
#        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
#        voting='soft')
#eclf2 = eclf2.fit(X_train, y_train)
#y_pred = eclf2.predict(X_test)
#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#
#eclf3 = VotingClassifier(estimators=[
#       ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
#       voting='soft', weights=[1,1,2],
#       flatten_transform=True)
#eclf3 = eclf3.fit(X_train, y_train)
#y_pred = eclf3.predict(X_test)
#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
