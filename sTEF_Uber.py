# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:59:21 2016

@author: Evelyn
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
matplotlib.style.use('ggplot')
import seaborn as sb
from datetime import datetime
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# load data
data = pd.read_csv('sTEF.csv')
# check amounts of variables and records
print data.shape
# check missing values
print np.sum(data.isnull())
'''
*******************************************************************************
1--Cleaning, exploration, visualization and preparation of data
*******************************************************************************
'''
#exempt records with missing bgc_date
data_bgc_done = data[data.bgc_date.notnull()]
# check dimension again
print data_bgc_done.shape

#exempt records with missing vehicle_added_date
data_vehicle_adding_done = data_bgc_done[data_bgc_done.vehicle_added_date.notnull()]
# check dimension again
print data_vehicle_adding_done.shape

#check missing values again
print np.sum(data_vehicle_adding_done.isnull())

#exclude id from predictive analysis
data_no_id = data_vehicle_adding_done.drop(['id'], 1)
#look at distributions of each predictor
data_no_id.city_name.value_counts().plot(kind='bar', color = 'green', title = 'City')
data_no_id.signup_os.value_counts().plot(kind='bar', color = 'green', title = 'signup_os')
data_no_id.signup_channel.value_counts().plot(kind='bar', color = 'green', title = 'signup_channel')

print data_no_id.vehicle_make.value_counts()
print data_no_id.vehicle_model.value_counts()
print data_no_id.vehicle_year.value_counts()

#convert first_completed_date into binary variable, 1-first_completed_date is not null, 0-otehrwise
data_no_id['first_completed_date'] = data_no_id['first_completed_date'].fillna(0)
data_no_id['first_completed_date'] = pd.to_datetime(data_no_id['first_completed_date'])
data_no_id.first_completed_date[data_no_id.first_completed_date > np.datetime64('1970-01-01')] = 1
data_no_id['first_completed_date'] = pd.to_numeric(data_no_id['first_completed_date'])

# plot relationship between predictor and dependent variable--turns out no statistical significance
sb.stripplot(x = "city_name", y = "first_completed_date", data = data_no_id)

# quick check ependency between Predictors and Dependent Variable
df_0_Strank = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['city_name'] == 'Strark')]
df_1_Strank = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['city_name'] == 'Strark')]

df_0_Berton = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['city_name'] == 'Berton')]
df_1_Berton = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['city_name'] == 'Berton')]

df_0_Wrouver = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['city_name'] == 'Wrouver')]
df_1_Wrouver = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['city_name'] == 'Wrouver')]

df_0_ios = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['signup_os'] == 'ios web')]
df_1_ios = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['signup_os'] == 'ios web')]

df_0_android = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['signup_os'] == 'android web')]
df_1_android = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['signup_os'] == 'android web')]

df_0_win = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['signup_os'] == 'windows')]
df_1_win = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['signup_os'] == 'windows')]

df_0_mac = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['signup_os'] == 'mac')]
df_1_mac = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['signup_os'] == 'mac')]

df_0_other = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['signup_os'] == 'other')]
df_1_other = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['signup_os'] == 'other')]

df_0_paid = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['signup_channel'] == 'Paid')]
df_1_paid = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['signup_channel'] == 'Paid')]

df_0_Referral = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['signup_channel'] == 'Referral')]
df_1_Referral = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['signup_channel'] == 'Referral')]

df_0_Organic = data_no_id[(data_no_id['first_completed_date'] == 0) & (data_no_id['signup_channel'] == 'Organic')]
df_1_Organic = data_no_id[(data_no_id['first_completed_date'] == 1) & (data_no_id['signup_channel'] == 'Organic')]

#casting signup_date, bgc_date and vehicle_added_date to datetime type
data_no_id['signup_date'] = pd.to_datetime(data_no_id['signup_date'])
data_no_id['bgc_date'] = pd.to_datetime(data_no_id['bgc_date'])
data_no_id['vehicle_added_date'] = pd.to_datetime(data_no_id['vehicle_added_date'])

# add intervals to data frame
data_no_id['signup_to_bgc'] = data_no_id['bgc_date'] - data_no_id['signup_date']
data_no_id['signup_to_vadd'] = data_no_id['vehicle_added_date'] - data_no_id['signup_date']
#data_no_id['bgc_to_vadd'] = data_no_id['vehicle_added_date'] - data_no_id['bgc_date']

#extract just days (int) from interval/timedelta64[ns]
data_no_id['signup_to_bgc'] = data_no_id['signup_to_bgc'].dt.days
data_no_id['signup_to_vadd'] = data_no_id['signup_to_vadd'].dt.days

#drop columns of three dates
data_no_dates = data_no_id.drop(['signup_date','bgc_date','vehicle_added_date'], 1)

#rename first_completed_date column
data_no_dates.rename(columns = {'first_completed_date': 'first_completed'}, inplace = True)

#fill all missing values in signup_os column with "other"
data_no_dates['signup_os'] = data_no_dates['signup_os'].fillna('other')

#******************************************************************************
#Method 1: create dummy variables for city_name. signup_os and signup_channel

#create dummy variables for categorical variables
data_dummy = pd.get_dummies(data_no_dates, columns = ['city_name','signup_os','signup_channel'])

#rearrange columns to move dependent variable to the end
print list(data_dummy)
data_rearranged = data_dummy[['vehicle_make','vehicle_model', 'vehicle_year', 'signup_to_bgc',
 'signup_to_vadd','city_name_Berton', 'city_name_Strark','city_name_Wrouver', 'signup_os_android web',
 'signup_os_ios web', 'signup_os_mac', 'signup_os_other', 'signup_os_windows', 'signup_channel_Organic',
 'signup_channel_Paid', 'signup_channel_Referral', 'first_completed']]
#******************************************************************************
'''
# Method 2: all categorical variables are encoded with LabelEncoder

data_rearranged = data_no_dates[['city_name', 'signup_os', 'signup_channel','vehicle_make','vehicle_model', 'vehicle_year', 'signup_to_bgc',
 'signup_to_vadd','first_completed']]

coder = preprocessing.LabelEncoder()
coder.fit(data_rearranged['city_name'])
data_rearranged['city_name'] = coder.transform(data_rearranged['city_name'])

coder = preprocessing.LabelEncoder()
coder.fit(data_rearranged['signup_os'])
data_rearranged['signup_os'] = coder.transform(data_rearranged['signup_os'])

coder = preprocessing.LabelEncoder()
coder.fit(data_rearranged['signup_channel'])
data_rearranged['signup_channel'] = coder.transform(data_rearranged['signup_channel'])
#******************************************************************************
'''

# Method 3: for categorical variabels with too many levels, combine levels for each variables

#calculate for each level the rate of response of 0 and 1
make0 = {}
make1 = {}
for i, make in enumerate(data_rearranged['vehicle_make']):
     make0[make] = (data_rearranged[(data_rearranged['first_completed'] == 0) & (data_rearranged['vehicle_make'] == make)]).shape[0]
     make1[make] = (data_rearranged[(data_rearranged['first_completed'] == 1) & (data_rearranged['vehicle_make'] == make)]).shape[0]

model0 = {}
model1 = {}
for i, model in enumerate(data_rearranged['vehicle_model']):
     model0[model] = (data_rearranged[(data_rearranged['first_completed'] == 0) & (data_rearranged['vehicle_model'] == model)]).shape[0]
     model1[model] = (data_rearranged[(data_rearranged['first_completed'] == 1) & (data_rearranged['vehicle_model'] == model)]).shape[0]

year0 = {}
year1 = {}
for i, year in enumerate(data_rearranged['vehicle_year']):
     year0[year] = (data_rearranged[(data_rearranged['first_completed'] == 0) & (data_rearranged['vehicle_year'] == year)]).shape[0]
     year1[year] = (data_rearranged[(data_rearranged['first_completed'] == 1) & (data_rearranged['vehicle_year'] == year)]).shape[0]

# calculate 0 rate of each level, try to combine levels with simlar response rates
make_rate = {}
for key in make0.keys():
    make_rate[key] = float(make0[key])/float(make0[key] + make1[key])

model_rate = {}
for key in model0.keys():
    model_rate[key] = float(model0[key])/float(model0[key] + model1[key])

year_rate = {}
for key in year0.keys():
    year_rate[key] = float(year0[key])/float(year0[key] + year1[key])

# combine levels with similar response rate
# make
for key in make0.keys():
    if make_rate[key] < 0.50:
        data_rearranged.vehicle_make[data_rearranged['vehicle_make'] == key] = 'L50'
    if (make_rate[key] < 0.55) & (make_rate[key] >= 0.50):
        data_rearranged.vehicle_make[data_rearranged['vehicle_make'] == key] = 'L55'
    if (make_rate[key] < 0.60) & (make_rate[key] >= 0.55):
        data_rearranged.vehicle_make[data_rearranged['vehicle_make'] == key] = 'L60'
    if (make_rate[key] < 0.65) & (make_rate[key] >= 0.60):
        data_rearranged.vehicle_make[data_rearranged['vehicle_make'] == key] = 'L65'
    if (make_rate[key] < 0.70) & (make_rate[key] >= 0.65):
        data_rearranged.vehicle_make[data_rearranged['vehicle_make'] == key] = 'L70'
    if (make_rate[key] < 0.75) & (make_rate[key] >= 0.70):
        data_rearranged.vehicle_make[data_rearranged['vehicle_make'] == key] = 'L75'
    if (make_rate[key] < 0.80) & (make_rate[key] >= 0.75):
        data_rearranged.vehicle_make[data_rearranged['vehicle_make'] == key] = 'L80'
    if make_rate[key] >= 0.80:
        data_rearranged.vehicle_make[data_rearranged['vehicle_make'] == key] = 'other'

# model
for key in model0.keys():
    if model_rate[key] < 0.10:
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M01'
    if (model_rate[key] < 0.20) & (model_rate[key] >= 0.10):
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M02'
    if (model_rate[key] < 0.30) & (model_rate[key] >= 0.20):
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M03'
    if (model_rate[key] < 0.40) & (model_rate[key] >= 0.30):
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M04'
    if (model_rate[key] < 0.50) & (model_rate[key] >= 0.40):
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M05'
    if (model_rate[key] < 0.60) & (model_rate[key] >= 0.50):
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M06'
    if (model_rate[key] < 0.70) & (model_rate[key] >= 0.60):
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M07'
    if (model_rate[key] < 0.80) & (model_rate[key] >= 0.70):
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M08'    
    if (model_rate[key] < 0.90) & (model_rate[key] >= 0.80):
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M09'        
    if model_rate[key] >= 0.90:
        data_rearranged.vehicle_model[data_rearranged['vehicle_model'] == key] = 'M10'

#year
#There are 3 records with vehicle_year = 0, we use the most frequent year "2015" to fill them.
data_rearranged.vehicle_year[data_rearranged['vehicle_year'] == 0] = 2015
#combine levels with low frequencies, like 1995~2000 and 2017
for key in year0.keys():
    if (key < 2001) & (key > 1994):
        data_rearranged.vehicle_year[data_rearranged['vehicle_year'] == key] = 2000
    if key == 2017:
        data_rearranged.vehicle_year[data_rearranged['vehicle_year'] == key] = 2000

#After level combination, create dummy variables for these categorical variables
data_no_make = data_rearranged.drop(['vehicle_make'], 1)
data_no_make = pd.get_dummies(data_no_make, columns = ['vehicle_model','vehicle_year'])
data_no_make = data_no_make[['signup_to_bgc', 'signup_to_vadd','city_name_Berton', 'city_name_Strark','city_name_Wrouver',
                             'signup_os_android web', 'signup_os_ios web', 'signup_os_mac', 'signup_os_other', 'signup_os_windows', 
                             'signup_channel_Organic', 'signup_channel_Paid', 'signup_channel_Referral', 
                             'vehicle_model_M01', 'vehicle_model_M02','vehicle_model_M03','vehicle_model_M04',
                             'vehicle_model_M05','vehicle_model_M06','vehicle_model_M07','vehicle_model_M08',
                             'vehicle_model_M09','vehicle_model_M10',
                             'vehicle_year_2000.0','vehicle_year_2001.0','vehicle_year_2002.0','vehicle_year_2003.0',
                             'vehicle_year_2004.0','vehicle_year_2005.0','vehicle_year_2006.0','vehicle_year_2007.0',
                             'vehicle_year_2008.0','vehicle_year_2009.0','vehicle_year_2010.0','vehicle_year_2011.0',
                             'vehicle_year_2012.0','vehicle_year_2013.0','vehicle_year_2014.0','vehicle_year_2015.0',
                             'vehicle_year_2016.0','first_completed']]
#******************************************************************************

#Data 1--drop vehicle model and formulate new data
data_no_model = data_rearranged.drop(['vehicle_model'], 1)
#encode vehicle_make and vehicle_year with LabelEncoder
make = preprocessing.LabelEncoder()
make.fit(data_no_model['vehicle_make'])
data_no_model['vehicle_make'] = make.transform(data_no_model['vehicle_make'])

year = preprocessing.LabelEncoder()
year.fit(data_no_model['vehicle_year'])
data_no_model['vehicle_year'] = year.transform(data_no_model['vehicle_year'])

#Data 2--drop vehicle make and formulate new data
data_no_make = data_rearranged.drop(['vehicle_make'], 1)
#encode vehicle_model and vehicle_year with LabelEncoder
model = preprocessing.LabelEncoder()
model.fit(data_no_make['vehicle_model'])
data_no_make['vehicle_model'] = model.transform(data_no_make['vehicle_model'])

year = preprocessing.LabelEncoder()
year.fit(data_no_make['vehicle_year'])
data_no_make['vehicle_year'] = year.transform(data_no_make['vehicle_year'])


'''
*******************************************************************************
2--Modeling, evaluation and comparison
*******************************************************************************
'''
#Logistic Regression
#Round1 data_no_model--Logstical Regression with 70% training set and 30% testting set
X_train, X_test, y_train, y_test = train_test_split(data_no_model.ix[:,:15], data_no_model['first_completed'], test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
predicted = model2.predict(X_test)
probs = model2.predict_proba(X_test)
# generate evaluation metrics
print metrics.accuracy_score(y_test, predicted)
print metrics.roc_auc_score(y_test, probs[:, 1])
#confusion matrix and a classification report with other metrics
print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)

#Logostical Regression with 10-fold cross validation
scores = cross_val_score(LogisticRegression(), data_no_model.ix[:,:15], data_no_model['first_completed'], scoring='accuracy', cv = 10)
print scores
print scores.mean()

#Round2 data_no_make--Logstical Regression with 70% training set and 30% testting set
X_train, X_test, y_train, y_test = train_test_split(data_no_make.ix[:,:15], data_no_make['first_completed'], test_size=0.3, random_state = 0)
model3 = LogisticRegression()
model3.fit(X_train, y_train)
predicted = model3.predict(X_test)
probs = model3.predict_proba(X_test)
# generate evaluation metrics
print metrics.accuracy_score(y_test, predicted)
print metrics.roc_auc_score(y_test, probs[:, 1])
#confusion matrix and a classification report with other metrics
print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)

#Logostical Regression with 10-fold cross validation
scores = cross_val_score(LogisticRegression(), data_no_make.ix[:,:15], data_no_make['first_completed'], scoring='accuracy', cv = 10)
print scores
print scores.mean()

#Round3 try to keep both vehicle model and make-Logstical Regression with 70% training set and 30% testting set
#encode vehicle_make, model and vehicle_year with LabelEncoder
make = preprocessing.LabelEncoder()
make.fit(data_rearranged['vehicle_make'])
data_rearranged['vehicle_make'] = make.transform(data_rearranged['vehicle_make'])

model = preprocessing.LabelEncoder()
model.fit(data_rearranged['vehicle_model'])
data_rearranged['vehicle_model'] = model.transform(data_rearranged['vehicle_model'])

year = preprocessing.LabelEncoder()
year.fit(data_rearranged['vehicle_year'])
data_rearranged['vehicle_year'] = year.transform(data_rearranged['vehicle_year'])

X_train, X_test, y_train, y_test = train_test_split(data_rearranged.ix[:,:16], data_rearranged['first_completed'], test_size = 0.3, random_state = 0)
model4 = LogisticRegression()
model4.fit(X_train, y_train)
predicted = model4.predict(X_test)
probs = model4.predict_proba(X_test)
# generate evaluation metrics
print metrics.accuracy_score(y_test, predicted)
print metrics.roc_auc_score(y_test, probs[:, 1])
#confusion matrix and a classification report with other metrics
print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)

#Logostical Regression with 10-fold cross validation
scores = cross_val_score(LogisticRegression(), data_rearranged.ix[:,:16], data_rearranged['first_completed'], scoring='accuracy', cv = 10)
print scores
print scores.mean()


#Random forest
#Round 1 on data without make

X_train, X_test, y_train, y_test = train_test_split(data_no_make.ix[:,:15], data_no_make['first_completed'], test_size = 0.1, random_state = 100)
model_RF = RandomForestClassifier(n_estimators = 1000)
model_RF.fit(X_train, y_train)
results = model_RF.predict(X_test)
print accuracy_score(y_test, results)

#plot the feature importance
importances = model_RF.feature_importances_
feature_importance = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align = 'center')
arr = np.array(list(data_no_make.columns.values))
plt.yticks(pos, arr[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
plt.show()

#Round 2 only use important features to predict
new_df = data_no_make[['signup_to_vadd', 'signup_to_bgc', 'vehicle_year', 'vehicle_model']]
X_train, X_test, y_train, y_test = train_test_split(new_df, data_no_make['first_completed'], test_size = 0.1, random_state = 100)
model_RF_new = RandomForestClassifier(n_estimators = 1000)
model_RF_new.fit(X_train, y_train)
results = model_RF_new.predict(X_test)
print accuracy_score(y_test, results)


#analyze how important features influence response
signup_to_bgc_0 = data_no_make.signup_to_bgc[data_no_make['first_completed'] == 0]
signup_to_bgc_1 = data_no_make.signup_to_bgc[data_no_make['first_completed'] == 1]

print signup_to_bgc_0.mean()
print signup_to_bgc_1.mean()

print signup_to_bgc_0.median()
print signup_to_bgc_1.median()

signup_to_vadd_0 = data_no_make.signup_to_vadd[data_no_make['first_completed'] == 0]
signup_to_vadd_1 = data_no_make.signup_to_vadd[data_no_make['first_completed'] == 1]

print signup_to_vadd_0.mean()
print signup_to_vadd_1.mean()

print signup_to_vadd_0.median()
print signup_to_vadd_1.median()


#GBM on date without make

X_train, X_test, y_train, y_test = train_test_split(data_no_make.ix[:,:7], data_no_make['first_completed'], test_size = 0.1, random_state = 100)
model_GBM = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 1)
model_GBM.fit(X_train, y_train)
results = model_GBM.predict(X_test)
print accuracy_score(y_test, results)
