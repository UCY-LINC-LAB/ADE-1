import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
def build_data_frame(file_name):
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
          				rows.append({"depth":features[0],"ratio":features[1],"hashtags":features[2],"length":features[3],"exla":features[4],"quest":features[5],"link":features[6],"topicRep":features[7],"users_d":features[8],"retweeted_d":features[9],"hashtags_d":features[10],"words_d":features[11],"lang_d":features[12],"neg":features[13],"neu":features[14],"pos":features[15],"compound":features[16],"category":0})
        			elif topic[1] == 'group interest':
          				rows.append({"depth":features[0],"ratio":features[1],"hashtags":features[2],"length":features[3],"exla":features[4],"quest":features[5],"link":features[6],"topicRep":features[7],"users_d":features[8],"retweeted_d":features[9],"hashtags_d":features[10],"words_d":features[11],"lang_d":features[12],"neg":features[13],"neu":features[14],"pos":features[15],"compound":features[16],"category":1})
        			elif topic[1] == 'news':
            				rows.append({"depth":features[0],"ratio":features[1],"hashtags":features[2],"length":features[3],"exla":features[4],"quest":features[5],"link":features[6],"topicRep":features[7],"users_d":features[8],"retweeted_d":features[9],"hashtags_d":features[10],"words_d":features[11],"lang_d":features[12],"neg":features[13],"neu":features[14],"pos":features[15],"compound":features[16],"category":2})
       				elif topic[1] == 'commemoratives':
            				rows.append({"depth":features[0],"ratio":features[1],"hashtags":features[2],"length":features[3],"exla":features[4],"quest":features[5],"link":features[6],"topicRep":features[7],"users_d":features[8],"retweeted_d":features[9],"hashtags_d":features[10],"words_d":features[11],"lang_d":features[12],"neg":features[13],"neu":features[14],"pos":features[15],"compound":features[16],"category":3})
					
				topics.append(topic[0])
		dataframe = DataFrame(rows,index=topics)
	return dataframe,topics

if __name__=='__main__':
	
	fs = 10  # fontsize
	
	data1,topics=build_data_frame('features.csv')
	labels = list('LGNC')
	#scaler = MinMaxScaler(feature_range=(0,10))
	
	scaler = Normalizer()
	data=scaler.transform(data1.astype(float))
	data = DataFrame(data1)
	data.columns = list(data1.columns.values)
	print data
	# demonstrate how to toggle the display of different elements:
	live_data1 = data.loc[data['category'] == 0]
	group_data1 = data.loc[data['category'] == 1]
	news_data1 = data.loc[data['category'] == 2]
	com_data1 = data.loc[data['category'] == 3]

	#l_data=live_data1["senti"]
	#g_data=group_data1["senti"]
	#n_data=news_data1["senti"]
	#c_data=com_data1["senti"]

	#normalize data
	#live_data1 = scaler.fit_transform(live_data1.astype(float))
	#group_data1 = scaler.fit_transform(group_data1.astype(float))
	#news_data1 = scaler.fit_transform(news_data1.astype(float))
	#com_data1 = scaler.fit_transform(com_data1.astype(float))	

	live_data = DataFrame(live_data1)
	live_data.columns = list(data.columns.values)

	group_data = DataFrame(group_data1)
	group_data.columns = list(data.columns.values)

	news_data = DataFrame(news_data1)
	news_data.columns = list(data.columns.values)


	com_data = DataFrame(com_data1)
	com_data.columns = list(data.columns.values)
	
	for feature in group_data[1:]:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.boxplot([live_data[feature].astype(float),group_data[feature].astype(float),news_data[feature].astype(float),com_data[feature].astype(float)])
		ax.set_title(feature, fontsize=fs)

	plt.show()






