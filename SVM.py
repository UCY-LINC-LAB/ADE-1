from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn import svm
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
def build_data_frame(file_name,cat):
	count_cat=0
	count=0
	with open(file_name,'rb') as tweets:	
		firstline=True
		topics=[]
		rows=[]
		for topic in tweets:
			if firstline:
				firstline=False
			else:
				topic=topic.strip().split('\t')
				features=topic[2:]
				if topic[1] == 'live events':
					if count<=count_cat or topic[1]==cat:
						if topic[1]==cat:
							count_cat+=1
	          					rows.append({"features":features,"category":1})
							topics.append(topic[0])
						else:
							count+=1
	          					rows.append({"features":features,"category":0})
							topics.append(topic[0])
        			elif topic[1] == 'group interest':
          				if count<=count_cat or topic[1]==cat :
						if topic[1]==cat:
							count_cat+=1
	          					rows.append({"features":features,"category":2})
							topics.append(topic[0])
						else:
							count+=1
	          					rows.append({"features":features,"category":0})
							topics.append(topic[0])
        			elif topic[1] == 'news':
            				if count<=count_cat or topic[1]==cat:
						if topic[1]==cat:
							count_cat+=1
	          					rows.append({"features":features,"category":3})
							topics.append(topic[0])
						else:
							count+=1
	          					rows.append({"features":features,"category":0})
							topics.append(topic[0])
       				elif topic[1] == 'commemoratives':
            				if count<=count_cat or topic[1]==cat:
						if topic[1]==cat:
							count_cat+=1
	          					rows.append({"features":features,"category":4})
							topics.append(topic[0])
						else:
							count+=1
	          					rows.append({"features":features,"category":0})
							topics.append(topic[0])
		dataframe = DataFrame(rows,index=topics)
	return dataframe

def build_data_frame1(file_name):
	with open(file_name,'rb') as tweets:	
		firstline=True
		firsttopic=True
		prevtopic=None
		flag=True
		topics=[]
		rows=[]
		for topic in tweets:
			print topic
			if firstline:
				firstline=False
			else:
				topic=topic.strip().split('\t')
				features=topic[2:]
				if topic[1] == 'live events':
	          			rows.append({"features":features,"category":1})
        			elif topic[1] == 'group interest':		
	          			rows.append({"features":features,"category":2})
        			elif topic[1] == 'news':
	          			rows.append({"features":features,"category":3})
       				elif topic[1] == 'commemoratives':
					rows.append({"features":features,"category":4})
				topics.append(topic[0])
				
		dataframe = DataFrame(rows,index=topics)
	return dataframe

if __name__=='__main__':

	data=build_data_frame1('features.csv')
	#data=data.sample(frac=1) #shuffle dataset
	feature = data['features'].values
	categories = data['category'].values
	#data=data.sample(frac=1) #shuffle dataset
	features=[]
	count=1
	for l in feature:
		l=[float(i) for i in l]
		features.append(l)
		count+=1
	features=pd.DataFrame(features)
	#normalize data
	

	#feature selection
	#test = SelectKBest(k=12)
	#fit = test.fit(features, categories)
	#features = fit.transform(features)
	clf=OneVsRestClassifier(svm.SVC(kernel='linear',gamma=0.001,C=100,max_iter=-1))
	#clf=OneVsRestClassifier(RandomForestClassifier(n_estimators=450,class_weight='balanced',min_samples_split=4))
	scores=cross_val_score(clf,features,categories,cv=5)
	print ("score: {:.2f}%".format(sum(scores)/len(scores)*100))

	
