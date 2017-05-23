#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

from nltk.stem.porter import PorterStemmer 

###############################################################################

X_train = []
y_train = []
target_names = ['live events', 'group interest', 'news', 'commemoratives']


with open('bb.csv','rb') as tweets:
    firstline=True
    stemmer=PorterStemmer()
    for tweet in tweets:
	line1=[]

	if firstline:
		firstline=False
	else:
        	fields = tweet.strip().split('\t')
       		text = fields[5]
		text = text.lower()
		text = re.sub(r"rt", "", text)
        	text = re.sub(r"http\S+", "", text)
		text = re.sub(r"([!.'():,?]+)", " ", text)
		text = re.sub(r"https\S+", "", text)
        	text = re.sub(r"(@[a-zA-Z0-9_]+)", "", text)
        	text = re.sub(r"([0-9]+)", "", text)
        	text = re.sub(r"(.)\1{1,}", r"\1\1", text)

        	if fields[1] == target_names[0]:
          		y_train.append(0)
        	elif fields[1] == target_names[1]:
          		y_train.append(1)
        	elif fields[1] == target_names[2]:
            		y_train.append(2)
       		elif fields[1] == target_names[3]:
            		y_train.append(3)
        	X_train.append(text)



clf = SGDClassifier(n_jobs = -1, n_iter = 100, eta0=0.1)


pipeline = Pipeline([
    ('vectorizer', HashingVectorizer(non_negative=True,n_features=(2 ** 18))),
    ('clf', CalibratedClassifierCV(base_estimator=clf, cv=5, method='isotonic'))
])

scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
print("score: {:.2f}%".format(sum(scores)/len(scores)*100))
print(scores)

