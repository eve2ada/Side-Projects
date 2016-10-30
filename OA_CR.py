# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:10:40 2016

@author: Evelyn
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from pandas.tools.plotting import scatter_matrix

'''
1. Load and summarize the included data
'''
# load data
data = pd.read_excel('ds_test_final.xls')
# check amounts of variables and records
print data.shape
# we get a data frame of 2215*7

# check missing values
print np.sum(data.isnull())
# so we have no missing values in this data set

print data.dtypes
# all columns are numeric

# generate summary statistics
print data.describe

'''
2. Determine a reasonable measure of model quality, 
and use the included data to compute and report this statistic.
3. Do you consider it a good model? Why or why not?
'''
# since the dependent variable is real valued
# we can use RMSE (root-mean-square error) to evaluete the model
evl_RMSE = (np.mean(data.Fitted_residuals ** 2))**0.5
# here we get the initial RMSE is 569.779
# Obviously the initial model is not good, since the RMSE is too large

'''
4. Discuss how well this model performs relative to a baseline model.
'''
# comparing with baseline model
# here we use use a central tendency measure as baseline, i.e.mean
# calculate baseline measurement
baseline_RMSE = (np.mean((data.Dependent_variable - 
                np.mean(data.Dependent_variable)) ** 2))**0.5
# Here we get the RMSE of baseline model: 615.459
# therefore the initial model performs worse than baseline model
# since we get smaller RMSE with baseline model

'''
5. Examine the columns of predictor variables. 
Note any predictors that should be transformed, 
dropped or interacted to improve the model.
'''
# build a Random Forest model to identify important factors
# just use default parameters
df_indep = data.ix[:, 3:]
#first build a linear regression model with all predictors-non-scaling
lm_1 = linear_model.LinearRegression()
lm_1.fit(df_indep, data["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((lm_1.predict(df_indep) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 613.898

#then scale all predictors
new_df = data.drop(["Fitted_residuals", "Fitted_Values"], 1)
min_max_scaler = preprocessing.MinMaxScaler()
new_df[['V6','V23','V34','V76']] = min_max_scaler.fit_transform(new_df[['V6','V23','V34','V76']])
lm_2 = linear_model.LinearRegression()
lm_2.fit(new_df.ix[:, 1:], new_df["Dependent_variable"])

#normalized_df_indep = preprocessing.normalize(new_df)
#lm_2 = linear_model.LinearRegression()
#lm_2.fit(normalized_df_indep.ix[:, 1:], normalized_df_indep["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((lm_2.predict(new_df.ix[:, 1:]) - data["Dependent_variable"]) ** 2))**0.5)
#here we get RMSE = 613.8905

# build Random Forest model
model_RF_1 = RandomForestRegressor()
model_RF_1.fit(df_indep, data["Dependent_variable"])

# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_1.predict(df_indep) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 276.4447

#produce importance of all predictors
importances = model_RF_1.feature_importances_
# here we get importance of 4 predictors are
#0.33823114,  0.3135912 ,  0.32208181,  0.02609585
#therefore we exclude the 4th predictor from our model
df_indep_2 = data.ix[:, 3:6]
model_RF_2 = RandomForestRegressor()
model_RF_2.fit(df_indep_2, data["Dependent_variable"])

# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_2.predict(df_indep_2) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 272.0874--a little bit improvement


#try other ways to improve model
df_new = df_indep
df_new["dependent"] = data["Dependent_variable"]
#generate scatter matrix to quickly check the dependency 
#of predictors and response variables
scatter_matrix(df_indep, alpha=0.2, figsize=(6, 6), diagonal='kde')
#It seems like V23 and V34 has kind of correlation

#then we add V23 * V34 as predictor
df_indep_3 = df_indep_2
df_indep_3["V23 * V34"] = df_indep_3["V23"] * df_indep_3["V34"]
model_RF_3 = RandomForestRegressor()
model_RF_3.fit(df_indep_3, data["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_3.predict(df_indep_3) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 279.8940--no improvement

# drop V34
df_indep_4 = df_indep_3.drop(['V34', 'V23 * V34'], 1)
model_RF_4 = RandomForestRegressor()
model_RF_4.fit(df_indep_4, data["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_4.predict(df_indep_4) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 287.9903--no improvement

# then we consider to add higher order effects to model
# add V6**2
df_indep_2 = data.ix[:, 3:6]
df_indep_5 = df_indep_2
df_indep_5["V6*V6"] = df_indep_5["V6"] * df_indep_5["V6"]
df_indep_5 = df_indep_5.drop(['V6'], 1)
model_RF_5 = RandomForestRegressor()
model_RF_5.fit(df_indep_5, data["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_5.predict(df_indep_5) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 281.828--no improvement

# add V23**2
df_indep_2 = data.ix[:, 3:6]
df_indep_6 = df_indep_2
df_indep_6["V23*V23"] = df_indep_6["V23"] * df_indep_6["V23"]
df_indep_6 = df_indep_6.drop(['V23'], 1)
model_RF_6 = RandomForestRegressor()
model_RF_6.fit(df_indep_6, data["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_6.predict(df_indep_6) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 279.3510

# add V34**2
df_indep_2 = data.ix[:, 3:6]
df_indep_7 = df_indep_2
df_indep_7["V34*V34"] = df_indep_7["V34"] * df_indep_7["V34"]
df_indep_7 = df_indep_7.drop(['V34'], 1)
model_RF_7 = RandomForestRegressor()
model_RF_7.fit(df_indep_7, data["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_7.predict(df_indep_7) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 273.3880

# if we tune the parameter of Random Forest
#then get some improvement
model_RF_8 = RandomForestRegressor(1000)
model_RF_8.fit(df_indep_7, data["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_8.predict(df_indep_7) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 237.9089

model_RF_9 = RandomForestRegressor(1000)
model_RF_9.fit(df_indep_6, data["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_9.predict(df_indep_6) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 237.1624

model_RF_10 = RandomForestRegressor(1000)
model_RF_10.fit(df_indep_5, data["Dependent_variable"])
# Calculate RMSE
print("Root mean squared error: %.4f"
      % (np.mean((model_RF_10.predict(df_indep_5) - data["Dependent_variable"]) ** 2))**0.5)
# here we get RMSE of 237.5949--no improvement

'''
Conclusion:
Due to limited time, we only tried several ways to select or transform
and improve the model.
By now the best model for our data set is Random Forest with below predictors:
V6, V34, V23**V23
The best RMSE is 237.1624
If we could know more about the predictors and get more records,
we can more improve the model.
'''