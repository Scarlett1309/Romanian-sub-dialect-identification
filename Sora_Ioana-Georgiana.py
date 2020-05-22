#!/usr/bin/env python
# coding: utf-8

# In[168]:


#Importing Libraries

import numpy as np
import pandas as pd
import seaborn as sn
import csv
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix, classification_report


# In[191]:


#Reading the files with pandas by using header = None, because the first line is interpreted as a header
#Train samples + labels -> the data we're using for training
train_samples = pd.read_fwf(r"train_samples.txt", header=None)
train_labels = pd.read_fwf(r"train_labels.txt", header = None)

#Validation samples + labels -> the data we're using for interpreting our accuracy, f1_score, etc
validation_samples = pd.read_fwf(r"validation_samples.txt", header = None)
validation_labels = pd.read_fwf(r"validation_labels.txt", header = None)

#Test samples -> the data we want to predict labels for
test_samples = pd.read_fwf(r"test_samples.txt", header = None)


# In[170]:


#Giving the fact that we are using DataFrames, which has multiple columns, we need to choose on which one to work
#We choose the second column, meaning, the column that has strings
#It's extracted as Series, but we transform it afterwards into an array

train_data = np.array(train_samples.iloc[:, 1])
train_data_label = np.array(train_labels.iloc[:, 1])

vld_data = np.array(validation_samples.iloc[:, 1])
vld_labels = np.array(validation_labels.iloc[:, 1])

test_data = np.array(test_samples.iloc[:, 1])


# In[171]:


#We're creating a class BagOfWords, given the fact that we can't work directly with strings
#We're creating a vocabulary of known words
#And also measure the presence of known words

class BagOfWords:
    #Class constructor
    def __init__(self):
        self.vocabulary = {} #Initializing an empty dictionary
        self.words = [] #Initializing an empty list

    #Method for creating a vocabulary using a given kind of data
    def build_vocabulary(self, train_data):
        for sentence in train_data:
            for word in sentence:
                #For each word in a sentence, but not found in the vocabulary, we're adding it to the dictionary
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.words)
                    self.words.append(word)
        return len(self.words)

    #Method for measuring the presence of known words
    def get_features(self, data):
        #Creating an empty matrix of sizes: nr of rows in data x length of the list
        result = np.zeros((data.shape[0], len(self.words)))
        for idx, sentence in enumerate(data):
            for word in sentence:
                if word in self.vocabulary:
                    result[idx, self.vocabulary[word]] += 1
        return result


# In[172]:


#Function to normalize the data, after creating our vocabulary
#We're using normalization because it improves the numerical stability of the model and often reduces training time
#We're considering multiple types of normalization : standard, MinMax or after L1/L2 distances

def normalize_data(train_data, validation_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1' or type == 'l2':
        scaler = preprocessing.Normalizer(norm=type)

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_validation_data = scaler.transform(validation_data)
        scaled_test_data = scaler.transform(test_data)
        return scaled_train_data, scaled_validation_data, scaled_test_data
    else:
        return train_data, validation_data, test_data


# In[173]:


#Using Bow to create our vocabulary for:
#1 Train Data
bow_model = BagOfWords()
bow_model.build_vocabulary(train_data)
training_features = bow_model.get_features(train_data)

#Validation Data
bow_model.build_vocabulary(vld_data)
vld_features = bow_model.get_features(vld_data)

#Test Data
bow_model.build_vocabulary(test_data)
test_features = bow_model.get_features(test_data)


# In[174]:


#Calling the previously defined method for normalizing the features obtained before
norm_train, norm_vld, norm_test = normalize_data(training_features, vld_features, test_features,type='l2')


# In[175]:
#Given the fact that the accuracy score improves when it has a greater number of training data,
#for our prediction we concatenate the train_data with validation data
#and also the train labels + validation labels

x_label = np.append(train_data_label, vld_labels)

x = np.append(train_data, vld_data)
bow_model.build_vocabulary(x)
x = bow_model.get_features(x)

scaler = None
scaler = preprocessing.Normalizer(norm='l2')
x_train = scaler.transform(x)
# In[176]:


#Training the model
clf = svm.SVC(kernel='rbf', C=1.5)
clf.fit(norm_train, train_data_label)


# In[177]:

#Accuracy score
test_preds = clf.predict(norm_vld)

print(f"Accuracy score : {accuracy_score(vld_labels, test_preds)}")


# In[178]:
#Classification report
print("Classification report : \n", classification_report(vld_labels, test_preds))


# In[179]:
#Confusion matrix

result = confusion_matrix(vld_labels, test_preds)
sn.heatmap(result, annot=True)
plt.show()


# In[180]:

#f1_score
f1_score(vld_labels, test_preds)

# In[ ]:


#Testing different data values for kernel / C and observing where the highest score is obtained
forPlot = []
b = []

clf = svm.SVC(kernel='rbf', C=1)
clf.fit(norm_train, train_data_label)
test_preds = clf.predict(norm_vld)
a = accuracy_score(vld_labels, test_preds)
forPlot.append(a)
b.append(1)

clf1 = svm.SVC(kernel='linear', C=1)
clf1.fit(norm_train, train_data_label)
test_preds = clf.predict(norm_vld)
a = accuracy_score(vld_labels, test_preds)
forPlot.append(a)
b.append(1)

clf1 = svm.SVC(kernel='linear', C=1.5)
clf1.fit(norm_train, train_data_label)
test_preds = clf.predict(norm_vld)
a = accuracy_score(vld_labels, test_preds)
forPlot.append(a)
b.append(1)

clf1 = svm.SVC(kernel='poly', C=1)
clf.fit(norm_train, train_data_label)
test_preds = clf.predict(norm_vld)
a = accuracy_score(vld_labels, test_preds)
forPlot.append(a)
b.append(1)

clf1 = svm.SVC(kernel='poly', C=1.5)
clf.fit(norm_train, train_data_label)
test_preds = clf.predict(norm_vld)
a = accuracy_score(vld_labels, test_preds)
forPlot.append(a)
b.append(1.5)


clf = svm.SVC(kernel='rbf', C=1)
clf.fit(norm_train, train_data_label)
test_preds = clf.predict(norm_vld)
a = accuracy_score(vld_labels, test_preds)
forPlot.append(a)
b.append(1)

clf = svm.SVC(kernel='rbf', C=1.5)
clf.fit(norm_train, train_data_label)
test_preds = clf.predict(norm_vld)
a = accuracy_score(vld_labels, test_preds)
forPlot.append(a)
b.append(1.5)


clf = svm.SVC(kernel='rbf', C=2)
clf.fit(norm_train, train_data_label)
test_preds = clf.predict(norm_vld)
a = accuracy_score(vld_labels, test_preds)
forPlot.append(a)
b.append(2)

plt.plot(forPlot, b, 'ro')
print(forPlot)


# In[181]:


#Prediction
clf = svm.SVC(kernel='rbf', C=1.5)
clf.fit(x_train, x_label)

prediction = clf.predict(norm_test)

prediction


# In[ ]:


#Creating the submission sample
x = np.array(test_samples.iloc[:, 0])
with open('filename.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["id", "Label"])
    for i in range(0, len(x)):
        wr.writerow([x[i], prediction[i]])
