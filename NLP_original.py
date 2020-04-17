#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:03:15 2019

@author: fahadtariq
"""

import pandas as pd
import matplotlib.pyplot as pt
import numpy as np

#importing the data
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#CLEANING THE TEXT
import re
import nltk
nltk.download('stopwords')   #nitk.download('all') //for downloading all kits.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpous = []
for i in range(0,1000):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) #removing colons,commas,dots,numbers etc
    review = review.lower() #all characters in lower case
    review = review.split() #for spliting each word
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpous.append(review)

#Creating the bag of words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpous).toarray()
y= dataset.iloc[:, 1].values

#Naive Bayes Classification

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting classifier to the Training set
# Create your classifier here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(55+91)/200