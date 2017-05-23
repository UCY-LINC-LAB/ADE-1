from pandas import DataFrame
import pandas as pd
import re
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
<<<<<<< HEAD
from nltk.stem.porter import PorterStemmer 

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
					rows2.append({"text":line,"category":fields[1]})
					topics2.append(fields[14])
				else :
					rows1.append({"text":line,"category":fields[1]})
					topics1.append(fields[14])
		dataframe1 = DataFrame(rows1,index=topics1)
		dataframe2 = DataFrame(rows2,index=topics2)
	return pd.concat([dataframe2,dataframe1])

if __name__=='__main__':
	data=build_data_frame('bb.csv')

	#data=data.groupby(['topics'],as_index=False).sum()
	#data = shuffle(data)
	print data
	#data=data.sample(frac=1) #shuffle dataset
	pipeline = Pipeline([	('vectorizer',CountVectorizer()),
				('tfidf', TfidfTransformer()),
				('classifier',MultinomialNB())
			   ])
	#Cross-Validating 
	kfold = KFold(n=len(data),n_folds=6)
	scores = []
	for train_ind,test_ind in kfold:
		train_text = data.iloc[train_ind]['text'].values
		train_y= data.iloc[train_ind]['category'].values

		test_text = data.iloc[test_ind]['text'].values
		test_y= data.iloc[test_ind]['category'].values
		
		pipeline.fit(train_text,train_y)
		predictions = pipeline.predict(test_text)
		
		score = accuracy_score(test_y, predictions)
		print score
		scores.append(score)

print ("Score: {:.2f}".format(sum(scores[1:])/len(scores[1:])*100))
