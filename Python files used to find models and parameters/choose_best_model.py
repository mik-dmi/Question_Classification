"""
Created on Sat Oct 24 16:46:02 2020

@author: me
"""
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB




def choose_best_model( X_train, y_train):

    '''
    max_features_range = np.arange(1,6,1)
    n_estimators_range = np.arange(10,210,10)
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

    rf =Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('random', RandomForestClassifier())])

    grid = GridSearchCV(estimator=rf['random'], param_grid=param_grid, cv=5)
    
    
    grid.fit(X_train, y_train)
    '''
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    
    
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
        GaussianNB(),
        ComplementNB(),
        BernoulliNB(),
        CategoricalNB()
        
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, X_train_tfidf, y_train, scoring='accuracy', cv=CV)
      #precision = precision_score(y_true, y_pred, labels=[1,2], average='micro')
      #print('Precision: %.3f' % precision)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    cv_df.groupby('model_name').accuracy.mean()
   # print(classification_report(y_test, y_pred))
    print(cv_df.groupby('model_name').accuracy.mean())
    
    
    
    
    
    
    
    