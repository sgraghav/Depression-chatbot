#
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
################################################################################################################################################
clf=joblib.load("training_set2.pkl")
string=raw_input('enter string:')
mood=raw_input('enter mood:')
string=review_to_words(string)
from sklearn.feature_selection  import SelectPercentile,f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=joblib.load("vectorizer2.pkl")
selector=joblib.load("selector2.pkl")
vector=vectorizer.fit_transform([string]).toarray()
#selector.fit(vector,[mood])
#vector=selector.transform([string])
clf.partial_fit(vector,[mood])
print("added to classifier")

