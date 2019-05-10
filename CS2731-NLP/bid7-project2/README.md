# bid7-hw2 README
HW2 Character Prediction


### Describe the computing environment you used, especially if you used some off-the-shelf modules. (Do not use unusual packages. If you’re not sure, please ask us.) 
* My code is written in Python, specifically python3. All of the packages I am using will be for Python3. There are few packages I am using but they are relatively well-known. In my code, I am using nltk, nltk's ngram, collections, collection's Counter, math, random, sklearn, torch, torch.nn, and time. Please ensure that these packages are installed before you run my code. To run, simply do "python3 character_predictor.py" without the quotes.


### List any additional resources, references, or web pages you've consulted.
* Only two outside resources were used. The first being an interpolation explanation from [Link](https://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf) and [Link](https://www.youtube.com/watch?v=Uj3iJbMfKYE)
* In terms of RNN tutorial, I have used pytorch and the tutorial I used is [Link](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)

### List any person with whom you've discussed the assignment and describe the nature of your discussions.
* I have talked to a friend named Zinan Zhuang about a specific problem. The problem was how to come up with an equation to generate lambdas for interoplation for any N to what I desired (explained on my REPORT) without hardcoding. He suggested that softmax may be of use to me and from there, I was able to come up with an equation to generate any number of N as needed. 


### Discuss any unresolved issues or problems. 
* Currently RNN does not work. Because RNN does not work, step 3 and 4 wasnt completed. However, I don't think there are any known bugs for steps 1 and 2. 

### How to run:
* Please ensure that the packages I listed up above are installed on your machine. The code is written in Python3 so please run my code on a machine that has Python3. 
* please ensure that the SFU_constructiveness.csv is in the same directory as the Python3 code.
* To run, simply do "python3 character_predictor.py" without the quotes.
* I have set up a small command line interface. Simply follow the directions outputted to terminal. 
