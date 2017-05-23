from pandas import DataFrame
import pandas as pd
import re
import numpy
from sklearn import svm
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import numpy as np
import math, re, string, requests, json
from itertools import product
from inspect import getsourcefile
from os.path import abspath, join, dirname
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier,LogisticRegression
from nltk.stem.porter import PorterStemmer 


common_dictionary=['a','for','what','like','me','you','we','do','have','had','did','who','how','good','fine','morning','night','now','too'
,'i','you','if','of','it','the','to','on','this','with','is','off','not','its','be','best','every','no','but','by','our','when','up','out','so'
,'my','more','from','is','are','in','that','does','where','could','us','just','can','thank','thanks','also','and','very','never','her','much']

#Create a dataframe which in the first column contains the text and in the second the category
def build_data_frame(file_name):
	stemmer= PorterStemmer()


	with open(file_name,'rb') as tweets:	
		firstline=True
		firsttopic=True
		prevtopic=None
		flag=True
		topics1=[]
		topics2=[]
		rows1=[]
		rows2=[]
		list1=[]
		for tweet in tweets:
			line1=[]
			if firstline:
				firstline=False
			else:
				fields = tweet.strip().split('\t')
				features=fields[6:]
				if firsttopic:
					prevtopic=fields[14]
					firsttopic=False
      	 			line = fields[5]
		 	       	line = re.sub(r"http\S+", "", line)
				line = re.sub(r"([!.'():,?%]+)", " ", line)
				line = re.sub(r"https\S+", "", line)
		       		line = re.sub(r"(@[a-zA-Z0-9_]+)", "", line)
		       		line = re.sub(r"([0-9]+)", "", line)
		       		line = line.lower()
				line = line.replace("rt", "")
	  	     		line = re.sub(r"(.)\1{1,}", r"\1\1", line)

				if prevtopic != fields[14] and flag:
					flag=False
					prevtopic=fields[14]
				elif prevtopic != fields[14] and not flag:
					flag=True
					prevtopic=fields[14]
				if flag :
					rows2.append({"text":line,"category":fields[1],"features":features})
					topics2.append(fields[14])
				else :
					rows1.append({"text":line,"category":fields[1],"features":features})
					topics1.append(fields[14])
		dataframe1 = DataFrame(rows1,index=topics1)
		dataframe2 = DataFrame(rows2,index=topics2)
	return pd.concat([dataframe2,dataframe1])


def norm(val):
	if val==0:
		return 0
	elif val<0.25:
		return 1
	elif val<0.5:
		return 2
	elif val<0.75:
		return 3
	else :
		return 4

class ItemSelector(BaseEstimator, TransformerMixin):
	
	def __init__(self,key):
		self.key=key

	def fit(self,x,y=None):
		return self

	def transform(self, data):
		return data[self.key]

class Sentiment(BaseEstimator, TransformerMixin):
	
	
	def fit(self,x,y=None):
		return self

	def transform(self, data):
		#return [{'hash': float(fea[0]),'url': float(fea[1]),'retweeted': float(fea[2]),'retweet': float(fea[4]),'favorite': float(fea[5]),'neg': float(fea[9]),'neu': float(fea[10]),'pos': float(fea[11]),'compound': float(fea[12])}for fea in data]
<<<<<<< HEAD
		return [{'hash': float(fea[0]),'url': float(fea[1]),'neg':float(fea[9]),'neu': float(fea[10][:4]),'pos': float(fea[11][:4]),'exla':float(fea[13]),'quest':float(fea[14])}for fea in data]


	


if __name__=='__main__':
	data=build_data_frame('data.csv')
	#clf=SGDClassifier(n_jobs = -1, n_iter = 100, eta0=0.1)
	#clf=OneVsRestClassifier(svm.SVC(kernel='rbf',gamma=0.001,C=100,max_iter=-1))
	clf=MultinomialNB()

	pipeline = Pipeline([	
			('features', FeatureUnion(
				transformer_list=[
					('sentiment',Pipeline([
						('selector',ItemSelector(key='features')),#Select numerical values
						('sentiment',Sentiment()),
						('vect',DictVectorizer())		  #Transforms lists of feature-value mapping to vectors
					])),
					('text',Pipeline([
						('selector',ItemSelector(key='text')),    #Select text values

						('tfidf', TfidfVectorizer())      	  #countVectorizer followed by TfidfTransformer	
	  		]))
	
				],
			
			transformer_weights={
				'sentiment':0.9,
				'text':1,
				},
			)),
			('classifier',CalibratedClassifierCV(base_estimator=clf, cv=5, method='isotonic'))
	
		])
	
	#Cross-Validating 
	kfold = KFold(n=len(data),n_folds=6)
	scores = []
	
	for train_ind,test_ind in kfold:
		train_text = data.iloc[train_ind]
		train_y= data.iloc[train_ind]['category']

		test_text = data.iloc[test_ind]
		test_y= data.iloc[test_ind]['category']

		pipeline.fit(train_text,train_y)
		predictions = pipeline.predict(test_text)
		#predictions = pipeline.predict(test_text)
		#print predictions,predictions_proba,test_y
		score = accuracy_score(test_y, predictions)
		print score
		scores.append(score)
#	print scores

	print ("Score: {:.2f}".format(((sum(scores[1:])/len(scores[1:]))*100)))

	
