from nltk.stem import SnowballStemmer
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import re
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



def evaluation( data_test, nb ,y_test, y_pred ):
 

    conf_mat = 0
    print('accuracy %s' % accuracy_score(y_pred, y_test))   # This is only used is you have a data with the answers for the questions( IT CHECKS LABELS( PREDICT) WITH LABELS (TRUE LABELS))
    print(classification_report(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred  )
    print ( "s<czdvf" ,conf_mat)
   
    return y_test














DEV_labels = 'DEV-labels_coarse.txt' 

#train_file = 'TRAIN.txt'
#test_file = 'DEV_questions.txt'

def model_coarse ( train_file, test_file  ):

    
    
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
    
    '''    
    #******************************************************************************************************************************      
    with open(DEV_labels , 'r') as fp2:                         #process the data from the test (questions)
        for test_it in fp2:                               
            test_it = test_it.strip()
            test_labels.insert(len(test_labels), test_it)     # list with only labels coarse
  
    #******************************************************************************************************************************    
    '''   
    
    
    
    
    
    
    
    
    #########################_______Dealing_with_the_txt_file_______________###############################
    
    
    
    #########################_______Putting_in_a_Panda_Data_set_____________###############################
    # intialise data of lists.
        
    
    data = {'labels_coarse' : labels_coarse ,'labels_fine' : labels_fine, 'questions' : questions}
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
    y = data.labels_coarse
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    
    
    
    fold = StratifiedKFold(n_splits = 10,random_state = None)
    fold.get_n_splits(X, y)
    #nb = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),('tfidf', TfidfTransformer(  sublinear_tf= True)),('clf',('chi', SelectKBest(chi2, k=1000)),LinearSVC(penalty ='l2', dual=False, tol=1e-3)),])
    nb= Pipeline([('vect',TfidfVectorizer(ngram_range = (1,2),  sublinear_tf = True, smooth_idf  = True)),('chi', SelectKBest(chi2, k = 10000)),('clf', LinearSVC(C = 1.0, penalty='l2', max_iter = 10000, dual = False))])
        
    for train_i, test_i  in fold.split(X,y):
        X_train,X_test = X.iloc[train_i], X.iloc[test_i]
        y_train,y_test = y.iloc[train_i], y.loc[test_i]
        nb.fit(X_train, y_train)
        
            
    
    
    
    #########################_______Splitting(cross_validation)________________________________###############################
          
    
    
    ########---------------------------Code to print the values on a text file  ----------###################
    
    #y_test = data_test['test_labels']      
             
    final_test =  data_test.test_questions      #predict the result og nb model to the test questions

    
    y_pred = nb.predict(final_test)
    
        
    
    #with open('OLA.txt','w') as fp2:
       
        #np.savetxt('test_questions.txt', data['test_questions'],fmt='%s')
    
    if (test_file == 'DEV_questions.txt' ):
        #y_test = evaluation( data_test, nb ,y_test, y_pred)
        np.savetxt('developNUM-coarse.txt', y_pred,fmt='%s')
             
   
    if(test_file == 'TEST.txt'):
                  
        np.savetxt('testNUM-coarse.txt', y_pred ,fmt='%s ')   
    
        #return data , y_pred, X_test
        
    #with open(base_filename1 ,'w') as fp2:
       
     #   np.savetxt(base_filename1, y_test,fmt='%s')
    
    
    
    #######---------------------------Code to print the values on a text file ----------###################
    return 
