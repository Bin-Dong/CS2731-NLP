import nltk
from nltk import ngrams
from collections import Counter
import math
import random
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import time


class Ngram(object):

    def __init__(self, N):
        self.vocab_set = []
        self.N = N
        self.NGram_counts = {}
        self.NGram_Minus_One_counts = {}
        self.stopwords = []
        self.sentences = []
        self.map = {}
        self.vocab_length = 0

    def train(self, filename):
        counter = 1
        if self.N > 1:
            print("Training", str(self.N) + "gram")
            for line in open(filename): #For each line in the file
                if counter % 5000 == 0:
                    print("Training Iteration:", counter)
                counter = counter + 1
                sentence = line.rstrip('\n') #Stripping the new line
                tokens = [c for c in sentence]
                sentence = [item for item in tokens if item.lower() not in self.stopwords]
                for x in range(0, self.N-1):
                    sentence.insert(0,chr(187)) #The symbol "»" denotes the start of the string

                sentence.extend(chr(248)) #The symbol "ø" denotes the end of the String

                self.vocab_set.extend(sentence)
                ngs = ngrams(sentence, self.N)
                for ng in ngs:
                    ng = "".join(str(x) for x in ng) #Converting list to string

                    if ng in self.NGram_counts:
                        self.NGram_counts[ng] = self.NGram_counts[ng] + 1
                    else:
                        self.NGram_counts[ng] = 1

                    startWith = ng[:self.N-1]
                    if startWith not in self.map:
                        self.map[startWith] = [ng]
                    elif ng not in self.map[startWith]:
                        self.map[startWith].append(ng)
                   

                
                ngsMinusOne = ngrams(sentence, self.N-1)
                for ng in ngsMinusOne:
                    ng = "".join(str(x) for x in ng) #Converting list to string
                    if ng in self.NGram_Minus_One_counts:
                        self.NGram_Minus_One_counts[ng] = self.NGram_Minus_One_counts[ng] + 1
                    else:
                        self.NGram_Minus_One_counts[ng] = 1
        elif self.N == 1:
            print("Training Unigram")
            for line in open(filename): #For each line in the file
                if counter % 5000 == 0:
                    print("Training Iteration:", counter)

                counter = counter + 1

                sentence = line.rstrip('\n') #Stripping the new line

                tokens = [c for c in sentence]
                sentence = [item for item in tokens if item.lower() not in self.stopwords]
                sentence.insert(0,chr(187)) #The symbol "»" denotes the start of the string
                    
                sentence.extend(chr(248)) #The symbol "ø" denotes the end of the String
                self.vocab_set.extend(sentence)
                ngs = ngrams(sentence, self.N)
                for ng in ngs: #creating mapping
                    ng = "".join(str(x) for x in ng) #Converting list to string

                    if ng in self.NGram_counts:
                        self.NGram_counts[ng] = self.NGram_counts[ng] + 1
                    else:
                        self.NGram_counts[ng] = 1

                    startWith = ng
                    if startWith not in self.map:
                        self.map[startWith] = [ng]
                    elif ng not in self.map[startWith]:
                        self.map[startWith].append(ng)
                   
        self.vocab_length = len(self.vocab_set)
        self.vocab_set = set(self.vocab_set)
                    
    #Given a string S, predict the next character using NGram
    #For example, given the string "worl", using Trigram
    #We will be getting the max Prob(x | "rl")
    def predict_no_smoothing(self, sentence):
        if self.N > 1:
            while len(sentence) < (n-1): #checking if the given needs patting or not
                sentence = chr(187) + sentence

            indexFrom = len(sentence) - (self.N-1)
            sequence = sentence[indexFrom:]

            highestProbability = -1
            predictedSequence = ""
            if sequence not in self.map:
                predictedSequence = random.sample(self.vocab_set, 1)[0]
            else:
                for x in self.map[sequence]:
                    if (self.NGram_counts[x] / self.NGram_Minus_One_counts[sequence]) > highestProbability:
                        highestProbability = self.NGram_counts[x] / self.NGram_Minus_One_counts[sequence]
                        predictedSequence = x[-1]
            return predictedSequence
        elif self.N == 1:
            highestProbability = -1
            predictedSequence = ""
            for x in self.vocab_set:
                if (self.NGram_counts[x] / self.vocab_length) > highestProbability:
                    predictedSequence = x
                    highestProbability = (self.NGram_counts[x] / self.vocab_length)
            return predictedSequence


#Prediction using Inter. Smoothing 
#I.E., if n = 3, we have
#x1Trigram + x2Bigram + x3Unigram
#where x1 > x2 > x3 and x1+x2+x3 = 1
def predict_smoothing(sentence, Model_List, N):
    alphas = []
    n = N
    for x in range(0, n+2):
        alphas.append(None) #create an array of size n+1 where index 1 maps to the weight for unigram, 2 maps to weight for bigram, etc.
                            #alphas[0] maps to the uniform distribution which is 1/ len(vocab_set)

    summation = 0
    for x in range(n+1,0, -1):
        if x == 1:
            summation = summation + 0.001
        else:
            summation = summation + n**x

    for x in range(n+1, 0 , -1):
        if x == 1:
            alphas[x] = 0.001 / summation
        else:
            alphas[x] = (n**x)/summation

    alphas.pop(0) #Popping index 0 because index 0 contains None. This will shift the entire list down where
                  #Index 0 maps to the weights for 1/len(vocab_set), 1 maps to weight for unigram, 2 maps to weight for bigram


    #At this point in time, we have the weights created.


    while len(sentence) < (n-1): #checking if the given needs patting or not
        sentence = chr(187) + sentence


    highestProbability = -1
    predictedLetter = ""

    #Going through each model in the list to get the highest probability. Applying interoplation smoothing
    for letter in Model_List[1].vocab_set:
        currentProbability = alphas[0] * (1 / len(Model_List[1].vocab_set))
        for index in Model_List:
            model = Model_List[index]
            if index == 1:
                currentProbability = currentProbability + (alphas[index] * (model.NGram_counts[letter] / model.vocab_length)) 
            else:
                given = sentence[(len(sentence) - model.N) +1 :]
                find = given + letter
                if find not in model.NGram_counts or given not in model.NGram_Minus_One_counts:
                    continue
                else:
                    currentProbability = currentProbability + (alphas[index] * (model.NGram_counts[find] / model.NGram_Minus_One_counts[given])) 
        if currentProbability > highestProbability:

            highestProbability = currentProbability
            predictedLetter = letter

    return predictedLetter

def calculatePRFS(predicted, target):
        target_names = set(predicted + target)
        result = classification_report(target, predicted, target_names=target_names)
        return result

def no_smoothing_test(input_file_test, NGrams, n):
    total_correct = 0
    total = 0

    counter = 0
    predicted_list = []
    actual_list = []

    for line in open(input_file_test): #For each line in the file
        if counter % 5000 == 0:
            print("Counter is at", counter)

        counter = counter + 1
        s = ""
        sentence = line.rstrip('\n') #Stripping the new line
        for character_index in range(0, len(sentence)):
            s = s+sentence[character_index]
            predicted_character = NGrams[n].predict_no_smoothing(s)
            predicted_list.append(predicted_character)

            if character_index + 1 == len(sentence): #If you reach the end of the sentence
                actual_list.append(chr(248))
                if predicted_character == chr(248):
                    total_correct = total_correct + 1
                #print("predicted", predicted_character, "target is", chr(248), "result is", result)
            else:
                actual_list.append(sentence[character_index+1])
                if predicted_character == sentence[character_index+1]:
                    total_correct = total_correct + 1

                #print("predicted", predicted_character, "target is", sentence[character_index+1], "result is", result)
            
            total = total + 1
    return total_correct, total, predicted_list, actual_list #Returns # of correct, # of predictions made, a list of predictions, and a list of targetted values

def smoothing_test(input_file_test, NGrams, n):
    total_correct = 0
    total = 0

    counter = 0
    predicted_list = []
    actual_list = []
    for line in open(input_file_test): #For each line in the file
        if counter % 5000 == 0:
            print("Counter is at", counter)

        counter = counter + 1
        s = ""
        sentence = line.rstrip('\n') #Stripping the new line
        for character_index in range(0, len(sentence)):
            s = s+sentence[character_index]
            #print("string is:", s)
            predicted_character = predict_smoothing(s, NGrams, n)
            predicted_list.append(predicted_character)

            if character_index + 1 == len(sentence): #If you reach the end of the sentence
                actual_list.append(chr(248))
                if predicted_character == chr(248):
                    total_correct = total_correct + 1
                #print("predicted", predicted_character, "target is", chr(248), "result is", result)
            else:
                actual_list.append(sentence[character_index+1])
                if predicted_character == sentence[character_index+1]:
                    total_correct = total_correct + 1
            total = total + 1

    return total_correct, total, predicted_list, actual_list #Returns # of correct, # of predictions made, a list of predictions, and a list of targetted values



'''----------------------------- Below are code (functions) for RNN. Do note that it does not work-----------'''

def createWordList(filename):
    vocab_set = []
    stopwords = []
    sentence_list = []
    counter = 0
    for line in open(filename): #For each line in the file
        if counter % 5000 == 0:
            print("Counter is at", counter)
        counter = counter + 1
        sentence = line.rstrip('\n') #Stripping the new line
        tokens = [c for c in sentence]
        sentence = [item for item in tokens if item.lower() not in stopwords]
        vocab_set.extend(sentence)
        ng = "".join(str(x) for x in sentence) #Converting list to string
        sentence_list.append(ng)

    return sentence_list, set(vocab_set)

# Given any line, turn it into a list of vocab_set
def convert_line_to_tensor(line, vocab_size, vocab_string):
    tensor = torch.zeros(len(line), 1, vocab_size)
    for character in range(len(line)):
        letter = line[character]
        tensor[character][0][vocab_string.find(letter)] = 1
    return tensor

#Create a tensor that 
def convert_line_to_target_tensor(line, vocab_size, vocab_string):
    letter_indexes = [vocab_string.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(vocab_size - 1) # EOS. The reason for -1 is because if there are a total 
    return torch.LongTensor(letter_indexes)

# Getting a random sample
def getRandomTrainingSample(sentence_list, vocab_size, vocab_string):
    randomIndex = random.randint(0, len(sentence_list)-1)
    line = sentence_list[randomIndex]
    input_line_tensor = convert_line_to_tensor(line, vocab_size, vocab_string)
    target_line_tensor = convert_line_to_target_tensor(line, vocab_size, vocab_string)
    return input_line_tensor, target_line_tensor


class RNN(nn.Module): #RNN Model. 2 layer NN
    def __init__(self, input_size, history_size, output_size):
        super(RNN, self).__init__()
        self.history_size = history_size

        self.i2h = nn.Linear(input_size + history_size, history_size)
        self.i2o = nn.Linear(input_size + history_size, output_size)
        self.o2o = nn.Linear(history_size + output_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, history):
        input_combined = torch.cat((input, history), 1)
        history = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((history, output), 1)
        output = self.o2o(output_combined)
        output = self.softmax(output)
        return output, history

    def initHistory(self):
        return torch.zeros(1, self.history_size)

def getTopProbCharacter(output,vocab_set):
    top_n, top_i = output.topk(1)
    index = top_i[0].item()
    return vocab_set[index], index

def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1) #each target_tensor must be in an array
    history = rnn.initHistory() #initialize history

    rnn.zero_grad() #zeroing out gradient

    loss = 0

    #for each input in tensor, train it and update the weight. 
    for i in range(input_line_tensor.size(0)):
        output, history = rnn.forward(input_line_tensor[i], history)
        l = criterion(output, target_line_tensor[i])
        loss += l

    #step back to prepare weight training
    loss.backward()
    print("loss here", loss)

    #training weight
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0) #return the last prediction and the average loss

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

input_file_train = input("Please enter the file you would like to be trained on. e.g. english/train:\n")
n = int(input("Please enter an 'n' degree value. e.g. 2 for bigram, 3 for Trigram, etc.:\n"))

mode = int(input("Select a mode:\n1 for user input and next character prediction\n2 for testing on test file\n"))


# input_file_train = 'english/train'
# #input_file_test = 'english/dev'
# input_file_test = 'english/test'
# n = 7
#input_file = "english/test"
#NGramModel = Ngram(2)
#NGramModel.train(input_file)

print("Please Wait")
NGrams = {}
for x in range(1,n+1): #n+1 is exclusive
    model = Ngram(x)
    model.train(input_file_train)
    NGrams[x] = model

if mode == 2:
    input_file_test = input("Please enter the file you would like to be tested on. e.g. english/test or english/dev:\n")
    print("Performing No Smoothing Test using", input_file_train, "as training data and", input_file_test,"as testing data")
    total_correct, total, predicted_list, actual_list = no_smoothing_test(input_file_test, NGrams, n)
    print(calculatePRFS(predicted_list,actual_list))
    print("Total Correct:" , total_correct,"/ Total",total,"=", (total_correct/total))
            
    print("Performing Smoothing Test using", input_file_train, "as training data and", input_file_test,"as testing data")
    total_correct, total, predicted_list, actual_list = smoothing_test(input_file_test, NGrams, n)
    print(calculatePRFS(predicted_list,actual_list))
    print("Total Correct:", total_correct,"/ Total",total,"=", (total_correct/total))
elif mode == 1:
    mode = int(input("Select a mode:\n1 for unsmoothed prediction\n2 for smoothed prediction\n3 to quit\n"))
    while(True):
        if mode == 3:
            exit(0)
    
        line = input("Enter some text. Type 'Quit' to quit\n")
        if line == "Quit" or line == "quit":
            exit(0)

        if mode == 1:
            predictedChar = NGrams[n].predict_no_smoothing(line)
            if predictedChar == chr(248):
                predictedChar = "EOS (End Of Sentence)"
            print("Predicted: '" + predictedChar + "'")
        elif mode == 2:
            predictedChar = predict_smoothing(line, NGrams, n)
            if predictedChar == chr(248):
                predictedChar = "EOS (End Of Sentence)"
            print("Predicted: '" + predictedChar + "'")

    


'''Below are the code for RNN. It is not working 
sentence_list, vocab_set = createWordList(input_file_train)
vocab_set = list(vocab_set)
vocab_size = len(vocab_set) + 1 #+1 for EOS
vocab_string = "".join(str(x) for x in vocab_set) #Converting list to string

# test_string = "hello" + chr(248)


# print(convert_line_to_tensor(test_string,vocab_size, vocab_string))
# target_sensor = convert_line_to_target_tensor(test_string, vocab_size, vocab_string)

# input_tensor, target_tensor = getRandomTrainingSample(sentence_list, vocab_size, vocab_string)
# print(len(input_tensor))
history_size = 128
#rnn = RNN(vocab_size, history_size, vocab_size)

#input, target = getRandomTrainingSample(sentence_list, vocab_size, vocab_string)
# history =rnn.initHidden()

# output, next_hidden = rnn.forward(input[0], history)
# print(getTopProbCharacter(output, vocab_set))

criterion = nn.NLLLoss()

learning_rate = 0.001

rnn = RNN(vocab_size, history_size, vocab_size)

n_iters = 100
print_every = 1
plot_every = 1
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    input_tensor, target_tensor = getRandomTrainingSample(sentence_list, vocab_size, vocab_string)
    output, loss = train(input_tensor, target_tensor)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0
'''