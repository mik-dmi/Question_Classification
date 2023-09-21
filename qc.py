import sys
import model_coarse as m_c
import pandas as pd
from sklearn.naive_bayes import ComplementNB




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


from nltk.stem import SnowballStemmer
import pandas as pd

import numpy as np
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB



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


DEV_labels = 'DEV-labels.txt'                 # only needed for the inicial part








def evaluation( data_test, nb ,y_test, y_pred ):
 

    conf_mat = 0
    print('accuracy %s' % accuracy_score(y_pred, y_test))   # This is only used is you have a data with the answers for the questions( IT CHECKS LABELS( PREDICT) WITH LABELS (TRUE LABELS))
    print(classification_report(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred  )
    print ( "s<czdvf" ,conf_mat)
   
    return y_test























def model_fine ( train_file, test_file  ):

    
    
#########################_______Set_Stemmer_and_Stopwords_______________###############################    
       
        
    stemmer = SnowballStemmer("english")
    STOPWORDS = set(stopwords.words('english'))
    not_stopwords = {'out','you' ,'than', 'being', 'after', 'how','by', 'here','at','up','under','above','from','down','because', 'whom','now', 'whom', 'for', 'few','between','where','if','which','while','once','when', 'before','these', 'those', 'there','what',  'who', 'why','all' } 
    STOPWORDS = set([word for word in STOPWORDS if word not in not_stopwords])
        
        
#########################_______Set_Stemmer_and_Stopwords_______________###############################        
    
    
    
#########################_______Dealing_with_the_txt_file_______________###############################    
    labels_total = []
    questions = []
    labels_fine = []
    labels_coarse = []
    test_questions = []
    test_labels = []
    with open(train_file, 'r') as fp:                          #process the data from the training set
        
        for line in fp :
           x = line.strip().split(" ", 1)
           labels_total.insert(len(labels_total), x[0])          #label completo
           questions.insert(len(questions),x[1])              # Just  questions
                   
        for s in labels_total:      
            x = s.split(":",1)
            labels_coarse.insert(len(labels_coarse), x[0])     # list with only labels coarse
            labels_fine.insert(len(labels_fine), x[1]) 
                
        
    with open(test_file , 'r') as fp1:                         #process the data from the test (questions)
        for quest in fp1:                               
            test_questions.insert(len(test_questions), quest)
    
    
#******************************************************************************************************************************      
    """
    with open(DEV_labels , 'r') as fp2:                         #process the data from the test (questions)
        for test_it in fp2:                               
            test_it = test_it.strip()
            test_labels.insert(len(test_labels), test_it)     # list with only labels coarse
    """
#******************************************************************************************************************************    
    
    
    
    
    
    
    
    
    
#########################_______Dealing_with_the_txt_file_______________###############################
    
    
    
#########################_______Putting_in_a_Panda_Data_set_____________###############################
    # intialise data of lists.
        
    
    data = {'labels_coarse' : labels_coarse ,'labels_fine' : labels_fine , 'labels_total': labels_total, 'questions' : questions}
    #data_test = {'test_questions' :test_questions ,'test_labels' : test_labels}
    data_test = {'test_questions' :test_questions}
#print ( data )
     
    # Create DataFrame
    data = pd.DataFrame(data)
    data_test = pd.DataFrame(data_test)
     
#########################_______Putting_in_a_Panda_Data_set______________###############################
    
    
    
#########################_______Cleaning_data ______________###############################   
         
       
        
    data['questions'] = data['questions'].apply(lambda x:" ".join( [stemmer.stem(i) for i in re.sub("[^a-zA-Z0-9)]", " ", x).split() if  i not in STOPWORDS]).lower())              # questions _ for training
    data_test['test_questions'] = data_test['test_questions'].apply(lambda x:" ".join( [stemmer.stem(i) for i in re.sub("[^a-zA-Z0-9)]", " ", x).split() if  i not in STOPWORDS]).lower())    #teste questions
    
#########################_______Cleaning_data ______________###############################   
      
    
    
#########################_______Splitting(cross_validation) _______________________________###############################
    
    X = data.questions
    y = data.labels_total
    
    
    fold = StratifiedKFold(n_splits = 10,random_state = None)
    fold.get_n_splits(X, y)
    nb= Pipeline([('vectorizer', CountVectorizer( ngram_range = (1 , 2) , max_df = 0.8, min_df = 1 , vocabulary = None ,analyzer = 'word'   )),('tfidf', TfidfTransformer(norm = 'l2' , sublinear_tf = True, smooth_idf = True , use_idf = False  )),('classifier', MultinomialNB (alpha = 2,fit_prior = True, ))])
    
   
    for train_i, test_i  in fold.split(X,y):
        X_train,X_test = X.iloc[train_i], X.iloc[test_i]
        y_train,y_test = y.iloc[train_i], y.loc[test_i]
        nb.fit(X_train, y_train)
        
       
    #y_test = data_test['test_labels']      
            
    final_test =  data_test.test_questions      #predict the result og nb model to the test questions

    
    y_pred = nb.predict(final_test) 
    
    
    
    
    
#########################_______Splitting(cross_validation)________________________________###############################
          
    
    
########---------------------------Code to print the values on a text file  ----------###################
    
    
    

    
    if (test_file == 'DEV_questions.txt' ):
        #y_test = evaluation( data_test, nb ,y_test, y_pred)
        np.savetxt('developNUM-fine.txt', y_pred,fmt='%s')
        
             
   
    if(test_file == 'TEST.txt'):
             
        
        np.savetxt('testNUM-fine.txt', y_pred ,fmt='%s ')
        
    
    
    
#######---------------------------Code to print the values on a text file ----------###################
    return data_test , y_pred, y_test , labels_total , data, test_labels,X_train, y_train










#######################################____MAIN______###################################

training_file_name = sys.argv[2] 
test_file = sys.argv[3]

if(len(sys.argv) == 4 ):
    if (sys.argv[1] == '-coarse'):
        m_c.model_coarse(training_file_name , test_file)   
        
    if (sys.argv[1] == '-fine'):
        data_test , y_pred, y_test , labels_total , data, test_labels,X_train, y_train = model_fine(training_file_name, test_file)                     
        
else:       
     print( "Wrong  arguments :\nOption1: 'python qc.py -coarse TRAIN.txt [name_of_test].txt' \nOption2: 'python qc.py -fine TRAIN.txt [name_of_test].txt'")
            

#######################################____MAIN______###################################



