#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:30:59 2019

@author: bread
"""

import csv
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer

""" Reads a .csv file and returns a list of strings and a list of labels. """
def read_csv(filename):
    """
    Input - .csv filename (filename)
    Output - List of strings (text)
           - List of labels (label)
    """
    text = []
    label = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        filereader = csv.reader(csvfile, delimiter="\t")
        next(filereader)
        for row in filereader:
            text.append(row[0])
            label.append(row[1])
    text, label = shuffle(text, label) # Shuffle the data
    return text, label

""" Creates a TF-IDF vector list given a string list"""
def TFIDF_Vectorization(text):
    """
    Input - A list of strings (text)
    Output - A list of TF-IDF vector
           - The vector itself
    """
    vectorizer = TfidfVectorizer(stop_words = None, 
                                ngram_range=(1,1), analyzer='word', 
                                tokenizer = None, preprocessor = None, 
                                token_pattern=r"(?u)\b\w+\b")
    tfidf = vectorizer.fit_transform(text)
    return vectorizer.transform(text).toarray().tolist(), vectorizer