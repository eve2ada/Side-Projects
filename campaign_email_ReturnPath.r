# Created on Sun Oct 9 10:58:21 2016
# Author: Evelyn

#make sure the data file is in current work directoty
# prepare the workspace
rm(list = ls())
set.seed(2016)

library(ggplot2)
library(car)
library(caret)
library(randomForest)

# load data and look at data
data_orig <- read.csv(file = "assessment_challenge.csv")
dim(data_orig)
colnames(data_orig)
str(data_orig)
head(data_orig)
summary(data_orig)
#pairs(data_orig)

##########################################################
# Data pre-processing
##########################################################

#exempt records with missing values
data_nomissing <- na.omit(data_orig)

summary(data_nomissing$from_domain_hash)
summary(data_nomissing$Domain_extension)
summary(data_nomissing$day)

#exclude id from predictive analysis
data_no_id <- subset(data_nomissing, select = -id)

# from here we construct two data sets for separate modeling

#one is excluding Domain_extension and another excluding from_domain_hash
data_no_domain <- subset(data_no_id, select = -Domain_extension)

# include new levels into the level space of from_domain_hash
new_levels <- 0:20
levels(data_no_domain$from_domain_hash) <- c(levels(data_no_domain$from_domain_hash), new_levels)

#Combine levels of from_domain_hash according to response rate
# run on 5% sample data set
temp <- sample(nrow(data_no_domain), round(0.05*nrow(data_no_domain)))
sample_hash <- data_no_domain[temp,]
training <- data_no_domain[-temp,]
#calculate group mean of response rate by from_domain_hash
mean_by_domain <-aggregate(sample_hash$read_rate, 
                            by = list(sample_hash$from_domain_hash),FUN = mean, na.rm = TRUE)

for(i in 1:length(mean_by_domain$Group.1)){
  sample_hash$from_domain_hash[sample_hash$from_domain_hash == mean_by_domain$Group.1[i]] <- as.integer(round(mean_by_domain$x[i], 2) * 100 /5) + 1
}

# drop levels with 0 records
sample_hash$from_domain_hash <- factor(sample_hash$from_domain_hash)
sample_hash <- na.omit(sample_hash)

# modeling and prediction
# training set & testing set separation--0.8 for training, 0.2 for testing
train <- sample(nrow(sample_hash), round(0.8*nrow(sample_hash)))
training_sample_no_domain <- sample_hash[train,]
testing_sample_no_domain <- sample_hash[-train,]

#   1. Linear regression model
model_lr_sample<-lm(read_rate ~ ., data = training_sample_no_domain)

# check the model performance
print(summary(model_lr_sample))
opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(model_lr_sample, las = 1)

#training_rmse_lr_1<-sqrt(mean((model_lr_sample$residuals)^2))
predictions<-predict(model_lr_sample, testing_sample_no_domain)
testing_rmse_lr_1 <- sqrt(mean((testing_sample_no_domain$read_rate - predictions)^2))

#   2. Random Forest
model_rf_sample <- randomForest(read_rate ~ ., data = training_sample_no_domain, 
                         ntree = 100, mtry = 10,
                         importance= T, na.action = na.omit)
print(summary(model_rf_sample))
plot(model_rf_sample)
importance(model_rf_sample)
varImpPlot(model_rf_sample)

#training_rmse_rf_sample <- sqrt(mean((training_sample_no_domain$read_rate - model_rf_sample$predicted)^2))
predictions<-predict(model_rf_sample, testing_sample_no_domain)
testing_rmse_rf_sample <- sqrt(mean((testing_sample_no_domain$read_rate - predictions)^2))


#3. Random Forest, try to train only with most important predictors
model_rf_selected_sample <- randomForest(read_rate ~ from_domain_hash 
                                  + avg_user_domain_avg_read_rate + avg_domain_inbox_rate + avg_user_avg_read_rate, 
                                  data = training_sample_no_domain, 
                                  ntree = 100, mtry = 5,
                                  importance= T, na.action = na.omit)
print(summary(model_rf_selected_sample))
plot(model_rf_selected_sample)
importance(model_rf_selected_sample)
varImpPlot(model_rf_selected_sample)

#training_rmse_rf_selected <- sqrt(mean((training$day_on_market - model_rf_selected$predicted)^2))
predictions<-predict(model_rf_selected_sample, testing_sample_no_domain)
testing_rmse_rf_selected_sample <- sqrt(mean((testing_sample_no_domain$read_rate - predictions)^2))



#Another is excluding from_domain_hash from data set
data_no_hash <- subset(data_no_id, select = -from_domain_hash)
#remove anomalies in Domain_extension
data_no_hash_done<-subset(data_no_hash, !grepl("[[:digit:]]", Domain_extension))

# include new levels into the level space of Domain_extension
new_levels <- 0:20
levels(data_no_hash_done$Domain_extension) <- c(levels(data_no_hash_done$Domain_extension), new_levels)

#Combine levels of Domain_extension according to response rate
#calculate group mean of response rate by Domain_extension
mean_by_hash <-aggregate(data_no_hash_done$read_rate, 
                           by = list(data_no_hash_done$Domain_extension),FUN = mean, na.rm = TRUE)

for(i in 1:length(mean_by_hash$Group.1)){
  data_no_hash_done$Domain_extension[data_no_hash_done$Domain_extension == mean_by_hash$Group.1[i]] <- as.integer(round(mean_by_hash$x[i], 2) * 100 /5) + 1
}

# drop levels with 0 records
data_no_hash_done$Domain_extension <- factor(data_no_hash_done$Domain_extension)

# modeling and prediction
# training set & testing set separation--0.8 for training, 0.2 for testing
train <- sample(nrow(data_no_hash_done), round(0.8*nrow(data_no_hash_done)))
training_sample_no_hash <- data_no_hash_done[train,]
testing_sample_no_hash <- data_no_hash_done[-train,]

#   1. Linear regression model
model_lr_no_hash<-lm(read_rate ~ ., data = training_sample_no_hash)

# check the model performance
print(summary(model_lr_no_hash))
#opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
#plot(model_lr_no_hash, las = 1)

#training_rmse_lr_1<-sqrt(mean((model_lr_sample$residuals)^2))
predictions<-predict(model_lr_no_hash, testing_sample_no_hash)
testing_rmse_lr_2 <- sqrt(mean((testing_sample_no_hash$read_rate - predictions)^2))

#   2. Random Forest
model_rf_no_hash <- randomForest(read_rate ~ ., data = training_sample_no_hash, 
                                ntree = 100, mtry = 10,
                                importance= T, na.action = na.omit)
print(summary(model_rf_no_hash))
plot(model_rf_no_hash)
importance(model_rf_no_hash)
varImpPlot(model_rf_no_hash)

#training_rmse_rf_sample <- sqrt(mean((training_sample_no_domain$read_rate - model_rf_sample$predicted)^2))
predictions<-predict(model_rf_no_hash, testing_sample_no_hash)
testing_rmse_rf_no_hash <- sqrt(mean((testing_sample_no_hash$read_rate - predictions)^2))


#3. Random Forest, try to train only with most important predictors
model_rf_selected_no_hash <- randomForest(read_rate ~ avg_user_domain_avg_read_rate + avg_domain_inbox_rate + avg_domain_read_rate + mb_superuser + avg_user_avg_read_rate, 
                                         data = training_sample_no_hash, 
                                         ntree = 100, mtry = 5,
                                         importance= T, na.action = na.omit)
print(summary(model_rf_selected_no_hash))
plot(model_rf_selected_no_hash)
importance(model_rf_selected_no_hash)
varImpPlot(model_rf_selected_no_hash)

predictions<-predict(model_rf_selected_no_hash, testing_sample_no_hash)
testing_rmse_rf_selected_sample <- sqrt(mean((testing_sample_no_hash$read_rate - predictions)^2))



