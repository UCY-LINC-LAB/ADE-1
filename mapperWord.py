#!/usr/bin/env python
import sys
import csv
import math
import string
import nltk
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer


csv.field_size_limit(sys.maxsize)
# input comes from STDIN (standard input)
stopwords=['.','!',';',',','\'',':','-','_','?','@','#','(',')','1','2','3','4','5','6','7','8','9','0','-','%','$','&','*','+','=','[',']']
dictionary=['a','for','what','like','me','you','we','do','have','had','did','who','how','good','fine','morning','night','now','too'
,'i','you','if','of','it','the','to','on','this','with','is','off','not','its','be','best','every','no','but','by','our','when','up','out','so'
,'my','more','from','is','are','in']
data = sys.stdin.readlines()
csvreader = csv.reader(data, delimiter='\t')
for row in csvreader:
	#Select category 
	# remove leading and trailing whitespace
	text = row[5].strip().decode('utf-8')
    	# split the line into words
	text = [ch.lower() for ch in text if ch not in stopwords]
	text=''.join(text).strip()
	text=text.split()
	# increase counters
	flag=0
	for word in text:
		flag=1
		# write the results to STDOUT (standard output);
		for stop in dictionary:
			if stop.lower() in word.lower():
				flag=0
		if  flag and len(word)>2:
			try:
        			print '%s %s\t%s' % (word.strip(),row[1], 1)
			except :
				continue

