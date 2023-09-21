from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import re
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from nltk.stem import SnowballStemmer
import choose_best_model as c_b_m
import tuning
#########################_______Pre-processing_programs_______________###############################    
   
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import KFold 

from sklearn.model_selection import StratifiedKFold

train_file = 'TRAIN.txt'


base_filename = 'Pred_Train_labels_coarse.txt'
base_filename1 = 'Test_Train_labels_coarse.txt'
train_file = 'TRAIN.txt'
nltk.download('wordnet')
# Init the Wordnet Lemmatizer



lemmatizer = WordNetLemmatizer()

   
stemmer=PorterStemmer()
STOPWORDS = set(stopwords.words('english'))
not_stopwords = {'out','you' ,'than', 'being', 'after', 'how','by', 'here','at','up','under','above','from','down','because', 'whom','now', 'whom', 'for', 'few','between','where','if','which','while','once','when', 'before','these', 'those', 'there','what',  'who', 'why','all' } 
STOPWORDS = set([word for word in STOPWORDS if word not in not_stopwords])
    
    
##########################_______Pre-processing_programs_______________###############################    



#########################_______Dealing_with_the_txt_file_______________###############################    
dict_initial = {}  # Labels and questions will be stored 
labels_total = []
questions = []
labels_fine = []
labels_coarse = []
    
 
    
with open(train_file, 'r') as fp:
    
    for line in fp :
        x = line.strip().split(" ", 1)
        labels_total.insert(len(labels_total), x[0])          #lable completo
        questions.insert(len(questions),x[1])              # Just the questions
               
    b = questions 
    for s in labels_total:      
        x = s.split(":",1)
        labels_coarse.insert(len(labels_coarse), x[0])     # list with only labels coarse
        labels_fine.insert(len(labels_fine), x[1])    

#########################_______Dealing_with_the_txt_file_______________###############################



#########################_______Putting_in_a_Panda_Data_set_____________###############################
# intialise data of lists.
    

data = {'labels_coarse':labels_coarse ,'labels_fine':labels_fine,'questions':questions, 'news_labels_coarse': None}
    
#print ( data )
 
# Create DataFrame
data = pd.DataFrame(data)

#########################_______Putting_in_a_Panda_Data_set______________###############################

    
    
     
data['questions'] = data['questions'].apply(lambda x: " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z0-9)]", " ", x).split() if  i not in STOPWORDS]).lower())
    
    




#########################_______Splitting _______________________________###############################

X = data.questions
y = data.labels_fine
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

#########################_______Splitting _______________________________###############################



# X is the feature set and y is the target




#########################_______Choosing_best_Model_and_Paremeters_______________________________###############################   
   
np.savetxt('sdfd.txt', y,fmt='%s')
#if( sys.argv[1] == 'models'):                     #find model 
#c_b_m.choose_best_model( X_train, y_train)
    
#if(sys.argv[1] == 'tuning'):                     #find paraemters for a model
tuning.tuning(X_train, y_train, X_test, y_test, X, y)


