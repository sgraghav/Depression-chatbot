import pandas
import os
import pickle
import re
import sys
import string
import numpy
import nltk
import csv
from sklearn.externals import joblib
from sklearn import cross_validation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer('english')
###############################################################################################################################################
#Stems passed string and returns stemmed string without special characters and punctuations
def review_to_words(raw_review):
      letters_only=re.sub("[^a-zA-Z]"," ",raw_review)
      words=letters_only.split()
      list=[]
      for k in words:
      	s=stemmer.stem(k)
        list.append(s)
      return(" ".join(list))
###############################################################################################################################################
"""
#training set 1
#word data is a list of string storing training statements....sentiid contains id to corresponding statement from word data
word_data=[]
sentiid=[]
with open('Tweets.csv','rb') as file:
      reader=csv.reader(file)
      i=0
      for row in reader:
          if row[1]=='airline_sentiment':
              continue
          i=i+1
          word_data.append(review_to_words(row[10]))
          sentiid.append(row[1])
          #if i==10000:
           #   break
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data,sentiid, test_size=0.1,random_state=42)
"""
###############################################################################################################################################
'''
#training set 2
features_train=[]
features_test=[]
labels_test=[]
labels_train=[]
file_train = open("training.txt","r")
test_file =open("Test-Data.txt","r")
for line in file_train:
	labels_train.append(line[0])
	features_train.append(review_to_words(line[1:]))

for line in test_file:
		labels_test.append(line[0])
		features_test.append(review_to_words(line[1:]))
'''
################################################################################################################################################
'''
#training set 3
#word data is a list of string storing training statements....sentiid contains id to corresponding statement from word data
word_data=[]
sentiid=[]
with open('Sentiment Analysis Dataset.csv','rb') as file:
      reader=csv.reader(file)
      i=0
   
      for row in reader:
          if row[1]=='Sentiment':
              continue
          i=i+1
          word_data.append(review_to_words(row[3]))
          sentiid.append(row[1])
          

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data,sentiid, test_size=0.2,random_state=42)
'''
################################################################################################################################################

#training set 4
features_train=[]
features_test=[]
labels_test=[]
labels_train=[]
word_data=[]
sentiid=[]
reader=pandas.read_csv('training.1600000.processed.noemoticon.csv',names=['1','2','3','4','5','6'],encoding='latin-1') 
print(reader.columns)
      #reader=csv.reader(file,encoding='latin-1')
for i in range(reader['1'].size):
    features_train.append(review_to_words(reader['6'][i]))
    labels_train.append(reader['1'][i])

reader=pandas.read_csv('testdata.manual.2009.06.14.csv',names=['1','2','3','4','5','6'],encoding='latin-1')
      #reader=csv.reader(file)
for i in range((reader['1'].size)):
    features_test.append(review_to_words(reader['6'][i]))
    labels_test.append(reader['1'][i])
'''import random
c = list(zip(features_train,labels_train))

random.shuffle(c)

features_train,labels_train= zip(*c)


features_train=features_train[:150000]
labels_train=labels_train[:150000]

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data,sentiid, test_size=0.2,random_state=42)'''
################################################################################################################################################
from sklearn.feature_selection  import SelectPercentile,f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(sublinear_tf=True,max_df=0.9,stop_words="english")

'''
def make_corpus(chunksize):
     chunkstartmarker = 0
     while chunkstartmarker < len(features_train):
        X_chunk=features_train[chunkstartmarker:chunkstartmarker+chunksize]
        yield X_chunk
        chunkstartmarker += chunksize
# list of files you want to load
corpus = make_corpus(len(features_train)/1000)
#rint(features_train)

#features_train_transformed=[]
'''
#esin_transformed=vectorizer.fit_transform(features_train).toarray()

features_train_transformed=vectorizer.fit_transform(features_train)
#print(features_train_transformed) 


features_test_transformed  = vectorizer.transform(features_test)
#features_test_transformed=features_test_transformed.toarray()
################################################################################################################################################
selector=SelectPercentile(f_classif,percentile=8)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed)
features_test_transformed  = selector.transform(features_test_transformed)
################################################################################################################################################
from sklearn.linear_model import SGDClassifier
def iter_minibatches(chunksize):
    # Provide chunks one by one
    chunkstartmarker = 0
    while chunkstartmarker < len(features_train_transformed):
        chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        X_chunk=features_train_transformed[chunkstartmarker:chunkstartmarker+chunksize]
        y_chunk=labels_train[chunkstartmarker:chunkstartmarker+chunksize]
        yield X_chunk, y_chunk
        chunkstartmarker += chunksize
 
batcherator = iter_minibatches(chunksize=len(features_train_transformed)/1000)
cld = SGDClassifier()
 
    # Train model
for X_chunk, y_chunk in batcherator:
    clf.partial_fit(X_chunk, y_chunk,classes=numpy.unique(labels_train))

################################################################################################################################################
#from sklearn.ensemble import RandomForestClassifier
#clf=RandomForestClassifier()
from sklearn.linear_model import SGDClassifier
clf=SGDClassifier()
clf.fit(features_train_transformed,labels_train)
#print(features_train_transformed)
#print(labels_train)
#print(labels_test)
pred=clf.predict(features_test_transformed)
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,labels_test))
joblib.dump(clf,"tr4bfclf.pkl")
joblib.dump(vectorizer,"tr4bfvec.pkl")
joblib.dump(selector,"tr4bfsel.pkl")

################################################################################################################################################





