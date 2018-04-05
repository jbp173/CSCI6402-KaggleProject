# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 01:49:36 2018

@author: Jordan Peters

The Kaggle submission requires prediction probabilities instead of straight classification predictions to calculate multi-class logarithmic loss. Ours was 0.65297.
"""
#Import the libraries that we're going to use
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm


#Read in training file
def readFile(file):
    f = open(file, encoding="utf8")
    lines = []
    for line in f:
        lines += [line]
    x, y = [], []
    for i in range (0, len(lines)):
        x += [lines[i][11:len(lines[i])-8]]
        y += [lines[i][len(lines[i])-5:len(lines[i])-2]]
    return x[1:len(x)], y[1:len(y)]

#Read in test file
def readTestFile(file):
    f = open(file, encoding="utf8")
    lines = []
    ids = []
    for line in f:
        lines += [line]
    x = []
    for i in range (0, len(lines)):
        x += [lines[i][11:len(lines[i])]]
        ids+=[lines[i][0:10]]
    return x[1:len(x)], ids[1:len(ids)]

xData, yData = readFile("train.csv")
trainX, ids = readTestFile("test.csv") #You can change this to draculaOnly.csv or otherBooks.csv to see what it thinks about other authors! Make sure to swap comments on a block 139-141 as well. 


features = []

#Feature extraction of training data
for sentence in xData:   
    feature = {}

    #Punctuation processing 
    if ";" in sentence:
        feature[";"] = 1
    if ":" in sentence:
        feature[":"] = 1 
    if "," in sentence:
        feature[","] = 1
    if "!" in sentence:
        feature["!"] = 1
    if "?" in sentence:
        feature["?"] = 1
            
    #Filter out all punctuation, since it will be attached to words and mess up our features
    sentence = re.sub('[^A-Za-z0-9 ]+', '', sentence.lower())
    
    
    #Count number of words in sentence
    words = sentence.split(" ")
    
    feature["len-"+str(len(words))] = 1
    
    
    #Each word is its own feature. If it's repeated, its a new feature 
    wordCount = {}
    for word in range(len(words)):
        if "word-"+words[word] not in wordCount:
            wordCount["word-"+words[word]] = 1
        else:
            wordCount["word-"+words[word]] += 1
            
        feature["word-"+words[word]+"-"+str(wordCount["word-"+words[word]])] = 1
        
        
    features += [feature]


vectorizer = DictVectorizer(sparse = True)      #Initialization of what will turn our feature arrays into sparse vectors
X = vectorizer.fit_transform(features)          #Actually make a sparse vector out of our training features we just made


print("start training") #Training takes a while, so it's good to know where you're at

clf=svm.SVC(kernel='linear',probability=True)    # use Linear SVM. This one allows for the probability vector, which is what the submission needs. 
clf.fit(X, yData)                                # train classifier

print("done training")

#Make predictions, output them to a new .csv file
output = open("spooky.csv","w+")

guess = ""
acc = 0
total = 0

pred = []
toOutput = ""

#Features are extracted in the same way, just on a different input
#Yeah, this probably could have been a function. This was easier since there was other processing I needed to do. 
for i  in range(len(trainX)):   
    feature = {}
    
    if ";" in trainX[i]:
        feature[";"] = 1
    if ":" in trainX[i]:
        feature[":"] = 1 
    if "," in trainX[i]:
        feature[","] = 1
    if "!" in trainX[i]:
        feature["!"] = 1
    if "?" in trainX[i]:
        feature["?"] = 1
            
    sentence = re.sub('[^A-Za-z0-9 ]+', '', trainX[i].lower())
    
    words = trainX[i].split(" ")    
    feature["len-"+str(len(words))] = 1
    
    
    wordCount = {}
    
    for word in range(len(words)):
        if "word-"+words[word] not in wordCount:
            wordCount["word-"+words[word]] = 1
        else:
            wordCount["word-"+words[word]] += 1
            
        feature["word-"+words[word]+"-"+str(wordCount["word-"+words[word]])] = 1
    
    #Create and format the output    
    pred = clf.predict_proba(vectorizer.transform([feature]))
    strPred = str(pred[0][0]) + "," + str(pred[0][1]) + "," + str(pred[0][2])
    toOutput += str(ids[i]) + "," + str(strPred) + "\n"
    
    #This is for if you just want to predict things. Use this instead of the above block for the other author files. 
    #pred = clf.predict(vectorizer.transform([feature]))
    #toOutput += str(pred) + "\n"

#Write output, close file, we're done!
output.write(str(toOutput))
output.close()
