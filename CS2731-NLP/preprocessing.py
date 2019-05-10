#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:09:40 2019

@author: Zane Denmon, Bin Dong, Dillon Schetley
"""

import sys
import csv
import re
from emoji import UNICODE_EMOJI
import shared_methods
from bs4 import BeautifulSoup
 
""" Strip all HTML tags """
def strip_html(inputString):
    """
    Input - .csv filename (filename)
    Output - List of strings (text)
           - List of labels (label)
    """
    return BeautifulSoup(inputString, "html.parser").text

""" Replaces any emojis in a string with space """
def deEmojify(inputString):
    returnString = ""
    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            # if emojis.__contains__(character):
            if character in UNICODE_EMOJI:
                # print('\nProblem char: ', character, '; unicode: ', character.encode('utf-8'), '\nWorking')
                returnString += ' '
            else:
                returnString += character
    return returnString

""" Preprocesses a string """
def preprocess(text):
    """
    Input - A string (text)
    Output - A preprocessed strings (text)
    """

    #Regex to remove URL and @ symbol
    regex = '@\S*|http\S*|www\S*'
    preprocessed_text = re.sub(regex, '', text)
    preprocessed_text = deEmojify(preprocessed_text)
    preprocessed_text = strip_html(preprocessed_text)

    return preprocessed_text
    
""" Writes values to a .csv file """
def write_csv(filename, text, labels):
    filename = filename.split(".")[0]
    filename = filename + "_preprocessed.csv"
    with open(filename, "w", encoding="utf-8", newline="\n") as csvfile:
        filewriter = csv.writer(csvfile, delimiter="\t")
        for t, l in zip(text, labels):
            filewriter.writerow([t,l])

def main():
    if (len(sys.argv) != 2):
        print("Please provide a .csv file as an argument.")
        sys.exit()
    filename = sys.argv[1]
    text, labels = shared_methods.read_csv(filename)
    text = list(map(lambda t: preprocess(t), text))
    text = ['text'] + text          # add back headers that get ignored during a read_csv() call
    labels = ['labels'] + labels    # add back headers that get ignored during a read_csv() call
    write_csv(filename, text, labels)


if __name__ == '__main__':
    main()