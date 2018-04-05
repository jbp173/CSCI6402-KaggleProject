# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 01:49:36 2018

@author: Jordan Peters  

Train and then evaluate a SVM Classifier or Neural Network for author identification between three novels.

The SVM outperforms the Neural Network by about double, with ~82% accuracy. 

This takes the last 2500 data points of the training data to test on. This is because the data is already randomized, 
and the test data file doesn't provide the actual classification, so we can't check our accuracy. 
"""
#Import the libraries that we're going to use to create the neural net and classifier
#Not all libraries will be used based on what model you run, but nothing needs to be commented out. 
import tensorflow as tf
import math #Python can't inherently do complex math, so we need external libraries to do sqaure root functions.
import re 
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm


#Read in training file
def readFile(file):
    f = open(file, encoding="utf8")
    lines = []
    ids = []
    for line in f:
        lines += [line]
    x, y = [], []
    for i in range (0, len(lines)):
        x += [lines[i][11:len(lines[i])-8]]
        y += [lines[i][len(lines[i])-5:len(lines[i])-2]]
        ids+=[lines[i][0:10]]
    return x[1:len(x)], y[1:len(y)]
    
xData, yData = readFile("train.csv")

'''
####Comment out for SVM#####
#We need one-hot vectors for the Neural Net, to make things easier
for i in range(len(yData)):
    if yData[i] == "MWS":
        yData[i] = [1,0,0]
    elif yData[i] == "EAP":
        yData[i] = [0,1,0]
    elif yData[i] == "HPL":
        yData[i] = [0,0,1]
'''     


#Feature extraction of training data
features = []

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

#'''
################SVM Classifier######################
#Split up data into train and test
trainX = features[:len(features)-2500]
trainY = yData[:len(yData)-2500]
testX = features[len(features)-2500:]
testY = yData[len(yData)-2500:]
        
vectorizer = DictVectorizer(sparse = True) #Initialization of what will turn our feature arrays into sparse vectors
X = vectorizer.fit_transform(trainX)       #Actually make a sparse vector out of our training features we just made
################END SVM Classifier######################
#'''



'''
############Neural Net################

vectorizer = DictVectorizer(sparse = False)
X = vectorizer.fit_transform(features)
trainX = X[:len(X)-2500]
trainY = yData[:len(yData)-2500]
testX = X[len(X)-2500:]
testY = yData[len(yData)-2500:]


#Neural network variables. These won't change during run time, but can be adjusted before you run to try to get better preformance. 
features = 28839                #Number of variables that we're looking at. This should match the number of features in the input files. As such, don't edit this for this data. Additionally, the number of input neurons. 
learning_rate = 0.005      #The rate at which the NN "learns"
training_steps = 500       #How many itterations though the data we should do
display_step = 50         #How often we should display our current progress
hidden1_units = 10          #The number of neurons in the first hidden layer
hidden2_units = 10         #The number of neurons in the second hidden layer

#Tensorflow requires you to create placeholders that you stream data into later on. 
x = tf.placeholder(dtype = tf.float32, shape = [None, features])    #We will stream our input features through here
y = tf.placeholder(dtype = tf.int32, shape = [None, 3])             #We will stream the correct classifications through here. It'll be a one-hot vector, so either [1,0] for no or [0,1] for yes. 

#This sets up the first hidden layer, which we connect to the inputs
with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([features, hidden1_units], stddev=1.0 / math.sqrt(float(features))),name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name = 'biases')
    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

#The second hidden layer, which we connect to the first
with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]), name = 'biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

#Our output layer, which is connected to the second hidden layer
with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, 3], stddev=1.0 / math.sqrt(float(hidden2_units))),name='weights')
    biases = tf.Variable(tf.zeros([3]), name='biases')
    logits = tf.matmul(hidden2, weights) + biases

#Cost and optimization functions
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#Evaluation functions
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Everything before this point was setup, now we'll start a Tensorflow session and actually loop through our data
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    
    #Loop through our data a number of times equal to what was previously specified
    for step in range(1, training_steps+1):
         
        #This is what's doing all the work here. This feeds in our training data through the network, which the net will attempt to classify.
        #It then compairs it to what the actual classification is, and will adjust its perameters accordingly.
        sess.run(optimizer, feed_dict={x: trainX, y: trainY})  
        
        #This will print out our accuracty however often we told it to, plus on the first itteration
        if step % display_step == 0 or step == 1:
            #Calculate batch loss and accuracy
            acc, loss = sess.run([accuracy, cost], feed_dict={x: trainX, y: trainY}) #This runs both the accuracy function and the cost function on the data at the same time, storing the results into the two seperate variables.
            print("Step " +str(step) + ", Batch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    #Finally, we let the user know we're done training, run our final net on the test data, and display our final results.
    #It's important to note that we're only testing the accuracy here, and not trying to furthur optimize the net.         
    print("Optimization Finished!")        
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testX, y: testY}))


################END Neural Net#############################
#'''


#'''
################################SVC CLassifier###########################################



clf = svm.LinearSVC()                             # use Linear SVM
clf.fit(X, trainY)                                # train classifier


guess = ""
acc = 0
total = 0

#Loop through all feature vectors in our test data and predict a classification for them
for i  in range(len(testX)):   
    
    pred = clf.predict(vectorizer.transform([testX[i]])) #Run the prediction function on the current feature vector
        
    #Accuracy Calculations. If we're right, add one to acc. Either way, add one to total.
    if pred == [testY[i]]:
        acc += 1              
    total += 1

print("Correct:", acc, "Total:", total, "Accuracy:", acc/total)


################################END SVC CLassifier###########################################
#'''
