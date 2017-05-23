from pandas import DataFrame
import pandas as pd
import re
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from nltk.stem.porter import PorterStemmer 
from pandas import DataFrame
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import math, re, string, requests, json
from itertools import product
from inspect import getsourcefile
from os.path import abspath, join, dirname
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier,LogisticRegression

#Create a dataframe which in the first column contains the text and in the second the category
def build_data_frame(file_name):
	stemmer=PorterStemmer()

	with open(file_name,'rb') as tweets:	
		firstline=True
		firsttopic=True
		prevtopic=None
		flag=True
		topics1=[]
		topics2=[]
		rows1=[]
		rows2=[]
		line1=[]
		for tweet in tweets:
			if firstline:
				firstline=False
			else:
				fields = tweet.strip().split('\t')
				features=fields[6:]
				if firsttopic:
					prevtopic=fields[14]
					firsttopic=False
      	 			line = fields[5]
				line = line.replace("rt", "")
				line = line.replace("RT", "")
		 	       	line = re.sub(r"http\S+", "", line)
				line = re.sub(r"([!.'():,?%]+)", " ", line)
				line = re.sub(r"https\S+", "", line)
		       		line = re.sub(r"(@[a-zA-Z0-9_]+)", "", line)
		       		line = re.sub(r"([0-9]+)", "", line)
		       		line = line.lower()
	  	     		line = re.sub(r"(.)\1{1,}", r"\1\1", line)
				line = line.strip()

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
		dataframe1 = DataFrame(rows1)
		dataframe2 = DataFrame(rows2)
	return pd.concat([dataframe2,dataframe1])

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
		return [{'hash': float(fea[0]),'url': float(fea[1]),'neg':float(fea[9]),'neu': float(fea[10][:4]),'pos': float(fea[11][:4]),'exla':float(fea[13]),'quest':float(fea[14])}for fea in data]

class Naive_output(BaseEstimator, TransformerMixin):
	
	
	def fit(self,x,y=None):
		return self

	def transform(self, data):
		return [{'c1': float(out[0]),'c2': float(out[1]),'c3': float(out[2]),'c4': float(out[3])}for out in data]



if __name__=='__main__':
	data=build_data_frame('bb.csv')

	#data=data.groupby(['topics'],as_index=False).sum()
	#data = shuffle(data)
	#data=data.sample(frac=1) #shuffle dataset
	pipeline = Pipeline([	('vectorizer',CountVectorizer()),
				('tfidf', TfidfTransformer()),
				('classifier',MultinomialNB())
			   ])
	#Cross-Validating 
	kfold = KFold(n=len(data),n_folds=6)
	scores = []
	prediction_NB=[]
	for train_ind,test_ind in kfold:
		train_text = data.iloc[train_ind]['text'].values
		train_y= data.iloc[train_ind]['category'].values

		test_text = data.iloc[test_ind]['text'].values
		test_y= data.iloc[test_ind]['category'].values
		
		pipeline.fit(train_text,train_y)
		predictions = pipeline.predict_proba(test_text)
		
		
		for probs in predictions:
			print probs
			prediction_NB.append({'NB_out':probs})
	
	
	prediction_NB=pd.DataFrame(prediction_NB)
	data = pd.concat([data,prediction_NB], axis=1, join='inner')
	clf1=OneVsRestClassifier(svm.SVC(kernel='rbf',gamma=0.001,C=100,max_iter=-1))
	#clf1=SGDClassifier(n_jobs = -1, n_iter = 100, eta0=0.1)
	print data
	pipeline1 = Pipeline([	
			('features', FeatureUnion(
				transformer_list=[
					('sentiment',Pipeline([
						('selector',ItemSelector(key='features')),#Select numerical values
						('sentiment',Sentiment()),
						('vect',DictVectorizer())		  #Transforms lists of feature-value mapping to vectors
					])),
					('NB_out',Pipeline([
						('selector',ItemSelector(key='NB_out')),#Select numerical values
						('sentiment',Naive_output()),
						('vect',DictVectorizer())		  #Transforms lists of feature-value mapping to vectors
					])),
					('text',Pipeline([
						('selector',ItemSelector(key='text')),    #Select text values
						('tfidf', TfidfVectorizer())      	  #countVectorizer followed by TfidfTransformer	
	  				]))
	
				],
			transformer_weights={
				'sentiment':0.6,
				'text':1,
				'NB_out':0.9,
				},
			
			)),
			('classifier',clf1)
	
			])
	#Cross-Validating 
	kfold = KFold(n=len(data),n_folds=5)
	scores = []
	
	for train_ind,test_ind in kfold:
		train_text1 = data.iloc[train_ind]
		train_y1= data.iloc[train_ind]['category']

		test_text1 = data.iloc[test_ind]
		test_y1= data.iloc[test_ind]['category']
		pipeline1.fit(train_text1,train_y1)
		predictions = pipeline1.predict(test_text1)
		score = accuracy_score(test_y1, predictions)
		print score
		scores.append(score)
#	print scores

	print ("Score: {:.2f}".format(((sum(scores[])/len(scores[]))*100)))

