import MySQLdb
import csv

if __name__ == '__main__':  
	
	conn = MySQLdb.connect(host= "localhost",user="root",passwd="px2h-r7s",db="twitter_data")
	x = conn.cursor()
	f=open("alltopics.csv")
	csv.reader(f)
	i=0
	for row in f:
		i=i+1
		text = row.replace('\n', ' ').replace('\r',' ')
		cat=text.strip().split("\t")
		try:
			x.execute("UPDATE `topics` SET `category`=\'"+cat[1]+"\' WHERE name = \""+cat[0]+"\"")
			conn.commit()	
			
		except:
			print "error"+str(cat)
			#print conn.rollback()
	f.close()
