# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:40:31 2016

@author: Evelyn
"""

# created by Evelyn on Tue Sep20 15:01 2016
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import pandas.io.sql
import pyodbc
import datetime as DT
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from time import time
from operator import itemgetter

# Define parameters
server = 'EVELYN-X200\SQLEXPRESS'
db = 'AdventureWorks2012'

# Create the connection
connection = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + db + ';Trusted_Connection=yes')

# query db
sql = """

SELECT  *  FROM HumanResources.Employee

"""

#load data to Python dataframe
df = pandas.io.sql.read_sql(sql, connection)
df.head()

#check dimensions of dataframe
print ("The dimensions of the dataframe we've got are "+ str(df.shape))

# check missing values
print np.sum(df.isnull())

# check types of all columns
print df.dtypes
'''
*******************************************************************************
1--clean and pre-process data
*******************************************************************************
'''

#pre-processing-1 identify unnecessary variables for classification

#according to the data dictionary(http://www.sqldatadictionary.com/AdventureWorks2012.pdf)
#and also check amount of unique values in each column to determine unnecessary variables
print len(df.BusinessEntityID.unique())
print len(df.NationalIDNumber.unique())
print len(df.LoginID.unique())

# convert hierarchyid to string values
df['OrganizationNode'] = df['OrganizationNode'].astype(str)
print len(df.OrganizationNode.unique())

print len(df.OrganizationLevel.unique())
print len(df.JobTitle.unique())
print len(df.BirthDate.unique())
print len(df.MaritalStatus.unique())
print len(df.Gender.unique())
print len(df.HireDate.unique())
print len(df.VacationHours.unique())
print len(df.SickLeaveHours.unique())
print len(df.CurrentFlag.unique())
print len(df.rowguid.unique())
print len(df.ModifiedDate.unique())

#drop columns with 290 unique values from classification
list_id = ["BusinessEntityID", "NationalIDNumber", "LoginID", 
                   "OrganizationNode", "rowguid"]
df_no_id = df.drop(list_id, 1)                 

#drop CurrentFlag since it has only one value
df_no_CurrentFlag = df_no_id.drop(['CurrentFlag'], 1)

#again check shape of dataset
print df_no_CurrentFlag.shape
print df_no_CurrentFlag.head()

#pre-processing-2 transform date time to duration

# get current date
now = pd.to_datetime(DT.datetime.now())
#calculate age and work_duration to substitute BirthDate and HireDate
df_no_CurrentFlag['Age'] = (now - pd.to_datetime(df_no_CurrentFlag['BirthDate'])).astype('<m8[Y]')
df_no_CurrentFlag['Work_Duration'] = (now - pd.to_datetime(df_no_CurrentFlag['HireDate'])).astype('<m8[Y]')
# also drop ModifiedDate since it provides limited info for classification
df_no_date = df_no_CurrentFlag.drop(['BirthDate', 'HireDate', 'ModifiedDate'], 1)

#pre-process-3 convert several non-numeric variables into category type
df_no_date['OrganizationLevel'] = df_no_date['OrganizationLevel'].astype('category')
df_no_date['JobTitle'] = df_no_date['JobTitle'].astype('category')
df_no_date['MaritalStatus'] = df_no_date['MaritalStatus'].astype('category')
df_no_date['Gender'] = df_no_date['Gender'].astype('category')

#pre-process-4 handling JobTitle(originally 67 levels)
#first check values distribution
print df_no_date['JobTitle'].value_counts()
#calculate for each level the rate of response of 0 and 1
flag0 = {}
flag1 = {}
for i, title in enumerate(df_no_date['JobTitle']):
     flag0[title] = (df_no_date[(df_no_date['SalariedFlag'] == 0) & (df_no_date['JobTitle'] == title)]).shape[0]
     flag1[title] = (df_no_date[(df_no_date['SalariedFlag'] == 1) & (df_no_date['JobTitle'] == title)]).shape[0]
# calculate 0 rate of each level
title_rate = {}
for key in flag0.keys():
    title_rate[key] = float(flag0[key])/float(flag0[key] + flag1[key])
# we can see every title must corresponds to 0 or 1
# if these records represent all possible titles, we may only use title to determine SalaryFlag
# but if there will be other titles besides these ones, we need to predict SalaryFlag with other predictors
# therefore here we drop JobTitle from classification model
df_no_title = df_no_date.drop(['JobTitle'], 1)

'''
# add new categories before combination
df_no_date.JobTitle = df_no_date.JobTitle.cat.add_categories(['SalaryFlag0', 'SalaryFlag1'])
for key in flag0.keys():
    if title_rate[key] == 1.0:
        df_no_date.JobTitle[df_no_date['JobTitle'] == key] = 'SalaryFlag0'
    else:
        df_no_date.JobTitle[df_no_date['JobTitle'] == key] = 'SalaryFlag1'
# remove unuesed categories
df_no_date.JobTitle = df_no_date.JobTitle.cat.remove_unused_categories() 
'''  

#pre-process-5 creat dummy variables for categorical variables
df_done = pd.get_dummies(df_no_title, columns = ['OrganizationLevel','MaritalStatus', 'Gender'])

'''
*******************************************************************************
2--Modeling, evaluation and comparison (default parameters)
*******************************************************************************
'''
#Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(df_done.ix[:, 1:], df_done['SalariedFlag'], 
                                                    test_size = 0.2, random_state = 0)
model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
predicted = model_LR.predict(X_test)
probs = model_LR.predict_proba(X_test)

# generate evaluation metrics
print metrics.accuracy_score(y_test, predicted)
print metrics.roc_auc_score(y_test, probs[:, 1])
#confusion matrix and a classification report with other metrics
print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)

#Logostical Regression with 10-fold cross validation
scores = cross_val_score(LogisticRegression(), df_done.ix[:, 1:], df_done['SalariedFlag'], 
                         scoring ='accuracy', cv = 10)
print scores
print scores.mean()

#Random Forest
model_RF = RandomForestClassifier(n_estimators = 1000)
model_RF.fit(X_train, y_train)
results = model_RF.predict(X_test)
print accuracy_score(y_test, results)

#plot the feature importance
importances = model_RF.feature_importances_
feature_importance = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
#plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align = 'center')
arr = np.array(list(df_done.columns.values)[1:])
plt.yticks(pos, arr[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
plt.show()

#Random Forest only using important features to predict
df_new = df_done[['OrganizationLevel_4', 'VacationHours', 'SickLeaveHours', 'Work_Duration', 'Age',
                  'OrganizationLevel_2', 'OrganizationLevel_3', 'OrganizationLevel_1']]
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(df_new, df_done['SalariedFlag'], 
                                                    test_size = 0.2, random_state = 100)
model_RF_new = RandomForestClassifier(n_estimators = 1000)
model_RF_new.fit(X_train_new, y_train_new)
results_new = model_RF_new.predict(X_test_new)
print accuracy_score(y_test_new, results_new)


'''
*******************************************************************************
3--Hyperparameter optimization using grid search
*******************************************************************************
'''
#get parameters list of classifier
print model_LR.get_params()
print model_RF_new.get_params()

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

#grid search for Logistic Regression
param_grid_LR = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'penalty': ['l1', 'l2']}
grid_search_LR = GridSearchCV(model_LR, param_grid_LR)
start = time()
grid_search_LR.fit(X_train, y_train)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search_LR.grid_scores_)))
report(grid_search_LR.grid_scores_)

test_performance_LR = grid_search_LR.score(X_test, y_test)
print test_performance_LR

#grid search for random forest
#specify parameters and distributions to sample from
param_grid_RF = {"max_features": ['auto', 'sqrt', 8],
              "min_samples_leaf": [1,2,3,4,5,6]}
grid_search_RF = GridSearchCV(model_RF_new, param_grid = param_grid_RF)
start = time()
grid_search_RF.fit(X_train_new, y_train_new)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search_RF.grid_scores_)))
report(grid_search_RF.grid_scores_)

#test_performance_RF = grid_search_RF.score(X_test_new, y_test_new)
#print test_performance_RF


#Train Random Forest model on the entire data set using the best hyperparameters found
#Random Forest only using important features to predict
model_RF_opt = RandomForestClassifier(n_estimators = 1000, 
                                      max_features = 'auto', min_samples_leaf = 1)
model_RF_opt.fit(df_new, df_done['SalariedFlag'])
results_opt = model_RF_opt.predict(df_new)
print accuracy_score(df_done['SalariedFlag'], results_opt)


#Random Forest using all predictors to predict
model_RF_opt_all = RandomForestClassifier(n_estimators = 1000, 
                                      min_samples_split = 3, min_samples_leaf = 1)
model_RF_opt_all.fit(df_done.ix[:, 1:], df_done['SalariedFlag'])
results_opt_all = model_RF_opt_all.predict(df_done.ix[:, 1:])
print accuracy_score(df_done['SalariedFlag'], results_opt_all)