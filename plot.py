print(__doc__)
import os
import pickle
import re
import sys
import string
import numpy as np
import nltk
import csv
import matplotlib.pyplot as plt
from sklearn import cross_validation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB
stemmer=SnowballStemmer('english')
###############################################################################################################################################
#Stems passed string and returns stemmed string without special characters
def review_to_words(raw_review):
      letters_only=re.sub("[^a-zA-Z]"," ",raw_review)
      words=letters_only.split()
      list=[]
      for k in words:
      	s=stemmer.stem(k)
        list.append(s)

      return(" ".join(list))
###############################################################################################################################################

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
          word_data.append((row[10]))
          if row[1]=="neutral":
              row[1]="positive"
          sentiid.append(row[1])
          #if i==10000:
           #   break


##########################################################################################################################################

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
          if i==6000:
              break
'''
###############################################################################################################################################
'''
#training set 
features=[]
labels=[]
file_train = open("training.txt","r")
for line in file_train:
        
	labels.append(line[0])
	features.append(review_to_words(line[1:]))
'''
################################################################################################################################################
'''from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,train_size=0.8,random_state=42)
features_train1,features_cv,labels_train1,labels_cv=train_test_split(features_train,labels_train,train_size=0.75,random_state=42)
'''
'''
from sklearn.feature_selection  import SelectPercentile,f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(sublinear_tf=True,max_df=0.5,stop_words="english")
features_train1_transformed = vectorizer.fit_transform(features_train1)
features_test_transformed  = vectorizer.transform(features_test)
features_cv_transformed=vectorizer.transform(features_cv)
features_cv_transformed=features_cv_transformed.toarray()
features_train1_transformed=features_train1_transformed.toarray()
features_test_transformed=features_test_transformed.toarray()
################################################################################################################################################
selector=SelectPercentile(f_classif,percentile=8)
selector.fit(features_train1_transformed, labels_train1)
features_train1_transformed = selector.transform(features_train1_transformed)
features_test_transformed  = selector.transform(features_test_transformed)
features_cv_transformed=selector.transform(features_cv_transformed)
selected_word_indices = selector.get_support(indices=True)
vocab = vectorizer.get_feature_names()
trimmed_vocab = [vocab[i] for i in selected_word_indices]
print[trimmed_vocab]
'''
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(sublinear_tf=True,max_df=0.5,stop_words="english")
features_transformed=vectorizer.fit_transform(word_data)
features_transformed=features_transformed.toarray()
selector=SelectPercentile(f_classif,percentile=20)
selector.fit(features_transformed,sentiid)
features_transformed=selector.transform(features_transformed)
##################################################################################################################################################
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

#estimator = GaussianNB()
#plot_learning_curve(estimator, title, features_transformed,sentiid, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, features_transformed,sentiid , (0.7, 1.01), cv=cv, n_jobs=4)

'''
title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
estimator = SVC(gamma=0.0019)
plot_learning_curve(estimator, title, features_transformed,labels, cv=cv, n_jobs=4)
'''
plt.show()

