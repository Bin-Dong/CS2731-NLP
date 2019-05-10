import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from nltk import ngrams
from sklearn.model_selection import cross_validate
import math
from scipy import stats
from sklearn.metrics import classification_report
from collections import Counter


def Unigram_extract_feature(bag_of_words, sentences):
    print("Unigram extracting...")
    feature_list = []
    for sentence in sentences:
        binary_feature = []
        for word in bag_of_words:
            if word in sentence:
                binary_feature.append(1)
            else:
                binary_feature.append(0)
        feature_list.append(binary_feature)
    return feature_list

def Bigram_extract_feature(bag_of_words, sentences):
    print("Bigram extracting...")
    feature_list = []
    for sentence in sentences:
        binary_feature = []
        sentence_bigram = ngrams(sentence, 2)
        sentence_bigram_list = list(sentence_bigram)
        for word in bag_of_words:
            if word in sentence_bigram_list:
                binary_feature.append(1)
            else:
                binary_feature.append(0)
        feature_list.append(binary_feature)
    return feature_list

def cross_validation(kf, Ngram_feature_list, y_label):
    print("Cross Valdating...")
    scoreResult = []
    i = 1
    for trained_index, test_index in kf.split(Ngram_feature_list):
        feature_list_test_x_label = []
        feature_list_test_y_label = []
        feature_list_train_x_label = []
        feature_list_train_y_label = []
        
        for index in trained_index:
            feature_list_train_x_label.append(Ngram_feature_list[index])
            feature_list_train_y_label.append(y_label[index])
        for index in test_index:
            feature_list_test_x_label.append(Ngram_feature_list[index])
            feature_list_test_y_label.append(y_label[index])
        clf = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
        clf.fit(feature_list_train_x_label ,feature_list_train_y_label)

        result = calculatePRFS(clf.predict(feature_list_test_x_label), feature_list_test_y_label)
        print("Fold:", i)
        print(result)
        print(" *** ")
        i = i + 1
        scoreResult.append(clf.score(feature_list_test_x_label, feature_list_test_y_label))


    return scoreResult

def baseline_cross_validation(kf, Ngram_feature_list, y_label):
    print("Cross Valdating...")
    scoreResult = []
    fold_i = 1
    for trained_index, test_index in kf.split(Ngram_feature_list):
        trained_label = []
        test_label = []
        
        for index in trained_index:
            trained_label.append(y_label[index])
        for index in test_index:
           test_label.append(y_label[index])


        numOnes = 0
        numTwos = 0
        numThrees = 0
        for x in range(0,len(trained_label)):
            if trained_label[x] == "1":
                numOnes = numOnes + 1
            elif trained_label[x] == "2":
                numTwos = numTwos + 1
            elif y_label[x] == "3":
                trained_label = numThrees + 1
        
        majority = None

        if numOnes >= numThrees and numOnes >= numThrees:
            majority = "1"
        elif numTwos >= numOnes and numTwos >= numThrees:
            majority = "2"
        elif numThrees >= numTwos and numThrees >= numOnes:
            majority = "3"
        else:
            print("Error in trying to figure out majority")


        numOnes = 0
        numTwos = 0
        numThrees = 0
        for x in range(0,len(test_label)):
            if test_label[x] == "1":
                numOnes = numOnes + 1
            elif test_label[x] == "2":
                numTwos = numTwos + 1
            elif y_label[x] == "3":
                test_label = numThrees + 1

        if majority == "1":
            scoreResult.append(numOnes/len(test_label))
        elif majority == "2":
            print("It is not 1!")
            scoreResult.append(numTwos/len(test_label))
        elif majority == "3":
            print("It is not 1!")
            scoreResult.append(numThrees/len(test_label))
        else:
            print("Error")

        predicted = []

        for i in range(0, len(test_label)):
            predicted.append(majority)
        
        result = calculatePRFS(predicted, test_label)
        print("Fold:", fold_i)
        print(result)
        print(" *** ")
        fold_i = fold_i + 1


    return scoreResult


def PValue_Calculate(unigram_score, bigram_score):
    result = stats.ttest_rel(unigram_score,bigram_score)
    return result

def calculatePRFS(y_pred, y_true):
        '''from sklearn.metrics import classification_report
            y_pred = [0, 2, 1,1,1]
            y_true = [0, 1, 2,1,0]
            target_names = ['class 0', 'class 1', 'class 2']
            print(classification_report(y_true, y_pred, target_names=target_names))
            #precision is basically number of times he got it correct for that specific / number of guesses for that specific class
            #recall = % where you can recall the number. Number of correctly recalled events / total events that exist
                        precision    recall  f1-score   support

            class 0       1.00      0.50      0.67         2
            class 1       0.33      0.50      0.40         2
            class 2       0.00      0.00      0.00         1

            micro avg     0.40      0.40      0.40         5
            macro avg     0.44      0.33      0.36         5
            weighted avg  0.53      0.40      0.43         5
        '''
        
        target_names = ['Label 1', 'Label 2']
        result = classification_report(y_true, y_pred, target_names=target_names)
        return result

def main():
    nltk.download('punkt')
    ###                                                               ###
    #   Reading CSV and Formatting Toxicity_level. Creating y_label     #
    ###                                                               ###
    input_file = "SFU_constructiveness.csv"
    csv = pd.read_csv(input_file)
    df = pd.DataFrame(data=csv)
    
    ###                                                                                                        ###
    #   Splitting the data before preprocessing into 80%/20% (80% trained and 20% will be final validation)      #
    #   The 20% will be articles titled:                                                                         #
    #   Is the B.C. property levy on 'foreign buyers' a new head tax? - The Globe and Mail                       #
    #                                           and                                                              #
    #   Thank you, Hillary. Now women know retreat is not an option - The Globe and Mail                         #
    #   Row number for those are from 828 to 1043. As a result, trained index = 0-826 and test index = 827-1042  #
    ###                                                                                                        ###


    y_label = []
    test_label = []
    for level in df['toxicity_level'][:827]:
        if level[0] == "4":
            y_label.append("2")
        elif level[0] == "3":
            y_label.append("2")
        else:
            y_label.append(level[0])

    for level in df['toxicity_level'][827:]:
        if level[0] == "4":
            test_label.append("2")
        elif level[0] == "3":
            test_label.append("2")
        else:
            test_label.append(level[0])

    ###                                                                                                      ###
    #   Creating tokenizing sentences, excluding stop words, creating bag_of_words, creating bigram_list.      #
    ###                                                                                                      ###
    stopwords = [',','.','!','a', 'an', 'and', 'are','as','at','be','by','for','from','has','he','in','is','it','its','of','on','that','the','to','was','were','with','will']
    sentences = []
    test_sentences = []
    bag_of_words = []
    bigram_bag_of_words = []
    for sentence in df['comment_text'][:827]:
        tokens = nltk.word_tokenize(sentence)
        sentences.append([item for item in tokens if item.lower() not in stopwords])
        bag_of_words.extend([item for item in tokens if item.lower() not in stopwords])
        
    for sentence in df['comment_text'][827:]:
        tokens = nltk.word_tokenize(sentence)
        test_sentences.append([item for item in tokens if item.lower() not in stopwords])
        

    bigram_dictionary = {}
    for sentence in sentences:
        bigrams = ngrams(sentence, 2)
        for bigram in bigrams:
            if bigram not in bigram_dictionary:
                bigram_dictionary[bigram] = 1
            else:
                bigram_dictionary[bigram] = bigram_dictionary[bigram] + 1
        #bigram_bag_of_words.extend(bigrams)

    bigram_bag_of_words = list(dict(Counter(bigram_dictionary).most_common(22500)))
    bag_of_words = set(bag_of_words)
    bigram_bag_of_words = set(bigram_bag_of_words)

    print(" ---------------------------------------------------------------- ")
    print("Extracting features for train dataset in both unigram and bigram")
    unigram_feature_list = Unigram_extract_feature(list(bag_of_words), sentences)
    bigram_feature_list = Bigram_extract_feature(list(bigram_bag_of_words), sentences)
    print(" ---------------------------------------------------------------- ")
    print("Extracting features for test dataset in both unigram and bigram")
    unigram_test_feature_list = Unigram_extract_feature(list(bag_of_words), test_sentences)
    bigram_test_feature_list = Bigram_extract_feature(list(bigram_bag_of_words), test_sentences)
    print(" ---------------------------------------------------------------- ")
    print("K Fold Testing")
    ###                 ###     
    #   KFold Testing.    #
    ###                 ###
    print("Begin KFold for Unigram:")
    #Unigram
    kf = KFold(n_splits = 10)
    scoreResult = cross_validation(kf, unigram_feature_list, y_label)

    avgScore = 0
    for score in scoreResult:
        avgScore = avgScore + score

    avgScore = avgScore / 10
    unigram_avgScore = avgScore
    unigram_scores = scoreResult
    print(" ---------------------------------------------------------------- ")
    
    print("Begin KFold for Bigram:")
    #Bigram
    kf = KFold(n_splits = 10)
    scoreResult = cross_validation(kf, bigram_feature_list, y_label)

    avgScore = 0
    for score in scoreResult:
        avgScore = avgScore + score

    avgScore = avgScore / 10
    bigram_scores = scoreResult
    bigram_avgScore = avgScore

    print(" ---------------------------------------------------------------- ")
    
    print("Begin KFold for Majority:")
    #Bigram
    kf = KFold(n_splits = 10)
    scoreResult = baseline_cross_validation(kf,bigram_feature_list, y_label)

    avgScore = 0
    for score in scoreResult:
        avgScore = avgScore + score

    avgScore = avgScore / 10
    majority_scores = scoreResult
    majority_avgScore = avgScore
    
    print("Cross validation complete. Results below:")
    print(" ---------------------------------------------------------------- ")
    print("KFold Accuracy Result for Unigram:", unigram_scores)
    print("KFold Accuracy Unigram Average Score:", unigram_avgScore)
    print("KFold Accuracy Result for Bigram:", bigram_scores)
    print("KFold Accuracy Bigram Average Score:", bigram_avgScore)
    print("KFold Accuracy Result for Majority:", majority_scores)
    print("KFold Accuracy Majority Average Score:", majority_avgScore)
    print(" ---------------------------------------------------------------- ")

    #Scoring on test data
    clf = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
    clf.fit(unigram_feature_list,y_label)
    print("Unigram Score on test data:", clf.score(unigram_test_feature_list, test_label))
    clf = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
    clf.fit(bigram_feature_list,y_label)
    print("Bigram Score on test data:", clf.score(bigram_test_feature_list, test_label))
    print(" ---------------------------------------------------------------- ")

    
    #Scoring on trained data
    clf = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
    clf.fit(unigram_feature_list,y_label)
    print("Unigram Score on trained data:", clf.score(unigram_feature_list, y_label))
    clf = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
    clf.fit(bigram_feature_list,y_label)
    print("Bigram Score on trained data:", clf.score(bigram_feature_list, y_label))
    print(" ---------------------------------------------------------------- ")
    



    ###                                                         ###
    #   Calculating Baseline for Majority for trained dataset     #
    ###                                                         ###

    numOnes = 0
    numTwos = 0
    numThrees = 0
    for x in range(0,len(y_label)):
        if y_label[x] == "1":
            numOnes = numOnes + 1
        elif y_label[x] == "2":
            numTwos = numTwos + 1
        elif y_label[x] == "3":
            numThrees = numThrees + 1
    
    majority = None

    if numOnes >= numThrees and numOnes >= numThrees:
        majority = "1"
    elif numTwos >= numOnes and numTwos >= numThrees:
        majority = "2"
    elif numThrees >= numTwos and numThrees >= numOnes:
        majority = "3"
    else:
        print("Error in trying to figure out majority")


    print("Majority Baseline for the 80 percent dataset that was used to train the model")
    if majority == "1":
        print("Majority is:", majority)
        print("Baseline Score is:", numOnes/len(y_label))
    elif majority == "2":
        print("Majority is:", majority)
        print("Baseline Score is:", numTwos/len(y_label))
    elif majority == "3":
        print("Majority is:", majority)
        print("Baseline Score is:", numThrees/len(y_label))
    else:
        print("Error")

    print(" ---------------------------------------------------------------- ")


    ###                                                         ###
    #   Calculating Baseline for Majority for test dataset        #
    ###                                                         ###

    numOnes = 0
    numTwos = 0
    numThrees = 0
    for x in range(0,len(test_label)):
        if test_label[x] == "1":
            numOnes = numOnes + 1
        elif test_label[x] == "2":
            numTwos = numTwos + 1
        elif test_label[x] == "3":
            numThrees = numThrees + 1
    
    majority = None

    if numOnes >= numThrees and numOnes >= numThrees:
        majority = "1"
    elif numTwos >= numOnes and numTwos >= numThrees:
        majority = "2"
    elif numThrees >= numTwos and numThrees >= numOnes:
        majority = "3"
    else:
        print("Error in trying to figure out majority")


    print("Majority Baseline for the 20 percent dataset that was used to test the model")
    if majority == "1":
        print("Majority is:", majority)
        print("Baseline Score is:", numOnes/len(test_label))
    elif majority == "2":
        print("Majority is:", majority)
        print("Baseline Score is:", numTwos/len(test_label))
    elif majority == "3":
        print("Majority is:", majority)
        print("Baseline Score is:", numThrees/len(test_label))
    else:
        print("Error")

    print(" ---------------------------------------------------------------- ")
    print("Performing T-Test vs. Unigram")
    PValue = PValue_Calculate(unigram_scores, bigram_scores)
    print("T-Test Results for Unigram vs Bigram", PValue)
    PValue = PValue_Calculate(unigram_scores, majority_scores)
    print("T-Test Results for Unigram vs Majority", PValue)
    print(" ---------------------------------------------------------------- ")
    print("Performing T-Test vs. Bigram")
    PValue = PValue_Calculate(bigram_scores, unigram_scores)
    print("T-Test Results for Bigram vs Unigram", PValue)
    PValue = PValue_Calculate(bigram_scores, majority_scores)
    print("T-Test Results for Bigram vs Majority", PValue)
    print(" ---------------------------------------------------------------- ")
    print("Performing T-Test vs. Majority")
    PValue = PValue_Calculate(majority_scores, unigram_scores)
    print("T-Test Results for Majority vs Unigram", PValue)
    PValue = PValue_Calculate(majority_scores, bigram_scores)
    print("T-Test Results for Majority vs Bigram", PValue)
    print(" ---------------------------------------------------------------- ")

    print("Calculating Precision, Recall, F1, and Support Score on trained dataset in unigram model")
    clf = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
    clf.fit(unigram_feature_list,y_label)
    result = calculatePRFS(clf.predict(unigram_feature_list), y_label)
    print(result)
    print(" ---------------------------------------------------------------- ")
    print("Calculating Precision, Recall, F1, and Support Score on test dataset in unigram model")
    clf = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
    clf.fit(unigram_feature_list,y_label)
    result = calculatePRFS(clf.predict(unigram_test_feature_list), test_label)
    print(result)
    print(" ---------------------------------------------------------------- ")
    print("Calculating Precision, Recall, F1, and Support Score on trained dataset in Bigram model")
    clf = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
    clf.fit(bigram_feature_list, y_label)
    result = calculatePRFS(clf.predict(bigram_feature_list), y_label)
    print(result)
    print(" ---------------------------------------------------------------- ")
    print("Calculating Precision, Recall, F1, and Support Score on test dataset in Bigram model")
    clf = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
    clf.fit(bigram_feature_list, y_label)
    result = calculatePRFS(clf.predict(bigram_test_feature_list), test_label)
    print(result)
    print(" ---------------------------------------------------------------- ")

    print("Experimenting Posed Question with Test Data")
    clf_unigram = LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')
    clf_bigram =  LogisticRegression(class_weight = "balanced", random_state=None, solver='lbfgs',multi_class='auto')

    clf_unigram.fit(unigram_feature_list, y_label)
    clf_bigram.fit(bigram_feature_list, y_label)

    result_unigram_prediction = clf_unigram.predict(unigram_test_feature_list)
    result_bigram_prediction = clf_bigram.predict(bigram_test_feature_list)

    majority_prediction = []
    for i in range(0, len(test_label)):
        majority_prediction.append(majority)
    
    majority_vote = []
    if len(majority_prediction) != len(result_unigram_prediction) and len(majority_prediction) != len(result_bigram_prediction):
        print("Something went wrong")
    else:
        for i in range(0, len(majority_prediction)):
            label = [0, 0]
            unigram_predict = int(result_unigram_prediction[i])
            bigram_predict = int(result_bigram_prediction[i])
            majority_predict = int(majority_prediction[i])

            label[unigram_predict - 1] = label[unigram_predict - 1] + 1
            label[bigram_predict - 1] = label[bigram_predict - 1] + 1
            label[majority_predict - 1] = label[majority_predict - 1] + 1

            if(label[0] > label[1]):
                majority_vote.append("1")
            else:
                majority_vote.append("2")

    howManyRight = 0
    totalNumber = len(majority_prediction)

    for i in range(0, totalNumber):
        if test_label[i] == majority_vote[i]:
            howManyRight = howManyRight + 1

    print("Accuracy for Majority Vote:", howManyRight/totalNumber)
    print(" ---------------------------------------------------------------- ")
    print("Calculating Precision, Recall, F1, and Support Score on Posed Question Experiment")
    result = calculatePRFS(majority_vote, test_label)
    print(result)
    print(" ---------------------------------------------------------------- ")

main()
