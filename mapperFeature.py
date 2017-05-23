#!/usr/bin/env python
import sys
import csv
import math
import string
import nltk
import re
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer


csv.field_size_limit(sys.maxsize)
# input comes from STDIN (standard input)
data = sys.stdin.readlines()
csvreader = csv.reader(data, delimiter='\t')
#each record
word=''
#exclude=['.','!',';',',','\'',':','-','_','?','@','RT','http','https','(',')','me','like','1','2','3','4','5','6','7','8','9','0','[',']','*']
for row in csvreader:
	# text
	text = row[5].strip().decode('utf-8')
	text1 = row[5].strip().decode('utf-8')
	text = text.lower()
        text = re.sub(r"http\S+", "", text)
	text = re.sub(r"([!.'():,]+)", " ", text)
	text = re.sub(r"https\S+", "", text)
        text = re.sub(r"(@[a-zA-Z0-9_]+)", "", text)
        text = re.sub(r"([0-9]+)", "", text)
        text = re.sub(r"(.)\1{1,}", r"\1\1", text)
	listText = text.split()
	# topic
	topic = row[0].strip().decode('utf-8')
	topic1 = row[0].strip().decode('utf-8')
	# category
	category = row[1].strip().decode('utf-8')
	# Retweet depth
	reDepth=0
	for word in listText:
		if word=="rt":
			reDepth+=1
	# Retweet ratio
	reRatio=0
	if "rt" in text:
		reRatio=1
	# Hashtag
	hashtags=row[6].strip()
	# Length
	length=len(text)
	# Exlamination
	exla=0
	if "!" in text1:
		exla=1
	# Question
	quest=0
	if "?" in text1:
		quest=1
	# Links
	link=row[7].strip()
	# Topic repetition
	topicRep=0
	topic1 = re.sub(r"([0-9]+)", "", topic1)
	topic1 = topic1.split()
	topic = topic.split()
	for word in listText:
		if word.lower()==topic1[0].lower() or word.lower()==topic[0].lower():
			topicRep+=1
	lang=row[9].strip().lower()
	topic=' '.join(topic)
	try:
		listText.remove('rt')
	except ValueError:
		print '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (topic,category,lang,listText,row[4],row[8],reDepth,reRatio,hashtags,length,exla,quest,link,topicRep,row[15],row[16],row[17],row[18])
		continue
	# increase counters
	#print text row[4]=userid row[8]=1 if is retweeted else 0
	
	print '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (topic,category,lang,listText,row[4],row[8],reDepth,reRatio,hashtags,length,exla,quest,link,topicRep,row[15],row[16],row[17],row[18])
	

