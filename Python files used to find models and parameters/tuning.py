# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:42:01 2020

@author: me
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from sklearn.naive_bayes import ComplementNB

base_filename = 'Pred_Train_labels_coarse.txt'
base_filename1 = 'Test_Train_labels_coarse.txt'
train_file = 'TRAIN.txt'



def tuning (X_train, y_train, X_test,y_test,X , y):
    
    
    nb = Pipeline([('vectorizer', CountVectorizer()),('tfidf', TfidfTransformer()), ('classifier',    ComplementNB())])



    nb.fit(X_train, y_train)
    
    
    
    
    
    scores = cross_val_score(nb, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1)
    
    mean = scores.mean()
    std = scores.std()
    print(mean)
    print(std)
    
    ''' 
    grid = {
        'vectorizer__analyzer' : ['word', 'char','char_wb'],
        'vectorizer__ngram_range' : [(1,1),(1,2),(1,3)],
        'vectorizer__max_df': [0.2,0.4,0.6,0.8,1],
        'vectorizer__min_df' : [0. ,0.2,0.4,0.5,0.6,0.8],
        'vectorizer__vocabulary' : [None , dict]
        
        
        #'tfidf__norm' : ['l1','l2'],
        #'tfidf__use_idf' : [True , False],
        #'tfidf__smooth_idf' : [True, False],
        #'tfidf__sublinear_tf' : [True]
        
        #'classifier__alpha' : [ 0.5 ,1,2,3,4,5,7],   
        #'classifier__fit_prior' : [True , False],
        #'classifier__norm' : [True, False] 
      
    }
    '''
    '''
    grid_search = GridSearchCV(nb, grid, scoring='accuracy', n_jobs=-1, cv=5)
    grid_search.fit(X=X_train, y=y_train)
    
    print("-----------")
    print(grid_search.best_score_)
    print(grid_search.best_params_)    
    
    df = pd.DataFrame(grid_search.cv_results_)
    print(df)
    
    
    
    
    
    #linearSVC
    grid = {
        
        
      'classifier__penalty': ('l1', 'l2'),
      'classifier__loss': ('hinge', 'squared_hinge'),
      'classifier__C': [0.1, 1, 10, 100, 1000],
           
      'classifier__max_iter' : [1000,2000 ,700 , 950, 10000]
    }
    
    grid_search = GridSearchCV(nb, grid, scoring='accuracy', n_jobs=-1, cv=5)
    grid_search.fit(X=X_train, y=y_train)
    
    print("-----------")
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    '''  
    
    
    pipeline2 = Pipeline([
        ('vectorizer', CountVectorizer( ngram_range = (1 , 2) ,  min_df=0, vocabulary = None ,analyzer = 'word'   )),
        ('tfidf', TfidfTransformer(norm = 'l2' , sublinear_tf = True, smooth_idf = True , use_idf = False  )),
        ('classifier', ComplementNB(alpha = 3 , fit_prior = True , norm = False))])
    
    model = nb.fit(X_train, y_train)
    model2 = pipeline2.fit(X_train, y_train)
    
    predicted = model.predict(X_test)
    predicted2 = model2.predict(X_test)
    
    print("model1: " + str(np.mean(predicted == y_test)))
    print("model2: " + str(np.mean(predicted2 == y_test)))