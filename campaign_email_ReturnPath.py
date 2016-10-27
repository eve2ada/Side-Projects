# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 12:05:19 2016

@author: Evelyn
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

# load data
data = pd.read_csv('assessment_challenge.csv')
# check amounts of variables and records
print data.shape
# check missing values
print np.sum(data.isnull())

'''
*******************************************************************************
1--Cleaning, exploration, visualization and preparation of data
*******************************************************************************
'''
#exempt records with missing values
data_nomissing = data.dropna()
# check dimension again
print data_nomissing.shape

#check types of all columns
print data_nomissing.dtypes

#then check levels of all categorical variables
print len(data_nomissing.from_domain_hash.unique())
print len(data_nomissing.Domain_extension.unique())
print len(data_nomissing.day.unique())

#exclude id from predictive analysis
data_no_id = data_nomissing.drop(['id'], 1)

# from here we construct two data sets for separate modeling
#one is excluding Domain_extension and another excluding from_domain_hash
data_no_domain = data_no_id.drop(["Domain_extension"], 1)


#Combine levels of from_domain_hash according to response rate
#calculate group mean of response rate by from_domain_hash
mean_by_domain = data_no_domain["read_rate"].groupby(data_no_domain["from_domain_hash"]).mean()
print len(mean_by_domain.unique())

#construct a dictionary for lookup
domain_rate = {}
for domain in mean_by_domain.index:
    domain_rate[domain] = mean_by_domain[domain]

#data_no_domain["from_domain_hash"] = int((domain_rate[data_no_domain["from_domain_hash"]].round(2))*100)/5 + 1
#data_no_domain["from_domain_hash"].apply(int((domain_rate[data_no_domain["from_domain_hash"]].round(2))*100)/5 + 1)

for domain in mean_by_domain.index:
    data_no_domain.from_domain_hash[data_no_domain["from_domain_hash"] == domain] = int((mean_by_domain[domain].round(2))*100)/5 + 1

print data_no_domain.head()
print data_no_domain.shape
print len(data_no_domain.from_domain_hash.unique())

data_no_domain['from_domain_hash'] = data_no_domain['from_domain_hash'].astype('category')
print data_no_domain.dtypes

#encode categorical variables of data_no_domain and day
domain = preprocessing.LabelEncoder()
domain.fit(data_no_domain['from_domain_hash'])
data_no_domain['from_domain_hash'] = domain.transform(data_no_domain['from_domain_hash'])

day = preprocessing.LabelEncoder()
day.fit(data_no_domain['day'])
data_no_domain['day'] = day.transform(data_no_domain['day'])

#Linear regression-data_no_domain
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(data_no_domain.ix[:, 1:], 
                                                            data_no_domain['read_rate'], test_size=0.2, random_state=0)
lm_no_domain = linear_model.LinearRegression()
lm_no_domain.fit(X_train_s, y_train_s)

# The coefficients
print('Coefficients: \n', lm_no_domain.coef_)
# The mean squared error
print("Mean squared error: %.4f"
      % np.mean((lm_no_domain.predict(X_test_s) - y_test_s) ** 2))

print("Root mean squared error: %.4f"
      % (np.mean((lm_no_domain.predict(X_test_s) - y_test_s) ** 2))**0.5)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % lm_no_domain.score(X_test_s, y_test_s))

#Random Forest for data_no_domain
model_RF_s = RandomForestRegressor(n_estimators = 1000)
model_RF_s.fit(X_train_s, y_train_s)
#results = model_RF_s.predict(X_test_s)

# The mean squared error
print("Mean squared error: %.4f"
      % np.mean((model_RF_s.predict(X_test_s) - y_test_s) ** 2))

print("Root mean squared error: %.4f"
      % (np.mean((model_RF_s.predict(X_test_s) - y_test_s) ** 2))**0.5)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % model_RF_s.score(X_test_s, y_test_s))

importances = model_RF_s.feature_importances_
feature_importance = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align = 'center')
arr = np.array(list(data_no_domain.columns.values)[1:])
plt.yticks(pos, arr[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
plt.show()

#Random Forest for data_no_domain only use important factors
X_test_new_s = data_no_domain[['from_domain_hash', 'avg_domain_inbox_rate', 'avg_user_avg_read_rate', 'avg_user_domain_avg_read_rate']]
X_train_new_s, X_test_new_s, y_train_new_s, y_test_new_s = train_test_split(X_test_new_s, data_no_domain['read_rate'], 
                                                    test_size = 0.2, random_state = 100)
model_RF_new_s = RandomForestRegressor(n_estimators = 1000)
model_RF_new_s.fit(X_train_new_s, y_train_new_s)

# The mean squared error
print("Mean squared error: %.4f"
      % np.mean((model_RF_new_s.predict(X_test_new_s) - y_test_new_s) ** 2))

print("Root mean squared error: %.4f"
      % (np.mean((model_RF_new_s.predict(X_test_new_s) - y_test_new_s) ** 2))**0.5)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % model_RF_new_s.score(X_test_new_s, y_test_new_s))

#another way; only keep Domain_extension
#remove anomalies in Domain_extension
data_no_digit = data_no_id[data_no_id.Domain_extension.str.isdigit() == False]
data_no_digit = data_no_digit[data_no_digit.Domain_extension.str.startswith("0") == False]
data_no_digit = data_no_digit[data_no_digit.Domain_extension.str.startswith("?") == False]
data_no_hash = data_no_digit.drop(["from_domain_hash"], 1)
print len(data_no_hash.Domain_extension.unique())
print data_no_hash.head()

#merge levels of Domain_extension based on response rate
mean_by_domain_name = data_no_hash["read_rate"].groupby(data_no_hash["Domain_extension"]).mean()
print len(mean_by_domain_name.unique())

for domain in mean_by_domain_name.index:
    data_no_hash.Domain_extension[data_no_hash["Domain_extension"] == domain] = int((mean_by_domain_name[domain].round(2))*100)/5 + 1

print data_no_hash.head()
print len(data_no_hash.Domain_extension.unique())
data_no_hash['Domain_extension'] = data_no_hash['Domain_extension'].astype('category')
print data_no_hash.dtypes

#encode categorical variables of data_no_hash
domain = preprocessing.LabelEncoder()
domain.fit(data_no_hash['Domain_extension'])
data_no_hash['Domain_extension'] = domain.transform(data_no_hash['Domain_extension'])

day = preprocessing.LabelEncoder()
day.fit(data_no_hash['day'])
data_no_hash['day'] = day.transform(data_no_hash['day'])

print data_no_hash.dtypes

#Linear regression-data_no_hash
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(data_no_hash.ix[:, 1:], data_no_hash['read_rate'], test_size=0.2, random_state=0)
lm_no_hash = linear_model.LinearRegression()
lm_no_hash.fit(X_train_t, y_train_t)
# The coefficients
print('Coefficients: \n', lm_no_hash.coef_)
# The mean squared error
print("Mean squared error: %.4f"
      % np.mean((lm_no_hash.predict(X_test_t) - y_test_t) ** 2))

print("Root mean squared error: %.4f"
      % (np.mean((lm_no_hash.predict(X_test_t) - y_test_t) ** 2))**0.5)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % lm_no_hash.score(X_test_t, y_test_t))

#Random Forest for data_no_hash
model_RF_t= RandomForestRegressor(n_estimators = 1000)
model_RF_t.fit(X_train_t, y_train_t)

# The mean squared error
print("Mean squared error: %.4f"
      % np.mean((model_RF_t.predict(X_test_t) - y_test_t) ** 2))

print("Root mean squared error: %.4f"
      % (np.mean((model_RF_t.predict(X_test_t) - y_test_t) ** 2))**0.5)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % model_RF_t.score(X_test_t, y_test_t))

importances = model_RF_t.feature_importances_
feature_importance = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align = 'center')
arr = np.array(list(data_no_hash.columns.values)[1:])
plt.yticks(pos, arr[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
plt.show()

#Random Forest for data_no_hash only use important factors
X_test_new_t = data_no_hash[['avg_domain_read_rate', 'avg_domain_inbox_rate', 'avg_user_avg_read_rate', 'avg_user_domain_avg_read_rate']]
X_train_new_t, X_test_new_t, y_train_new_t, y_test_new_t = train_test_split(X_test_new_t, data_no_hash['read_rate'], 
                                                    test_size = 0.2, random_state = 100)
model_RF_new_t = RandomForestRegressor(n_estimators = 1000)
model_RF_new_t.fit(X_train_new_t, y_train_new_t)

# The mean squared error
print("Mean squared error: %.4f"
      % np.mean((model_RF_new_t.predict(X_test_new_t) - y_test_new_t) ** 2))

print("Root mean squared error: %.4f"
      % (np.mean((model_RF_new_t.predict(X_test_new_t) - y_test_new_t) ** 2))**0.5)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % model_RF_new_t.score(X_test_new_t, y_test_new_t))



