# Created on Fri Sep 2 18:59:21 2016
# Author: Evelyn
# Please read before running this file:
  # Please make sure the data file ListingsAndSales.csv is located in current work directory.
  # Please make sure packages listed below have been installed: ggmap, ggplot2, caret, gbm, rondomForest,lme4, plyr
  # Since ggmap's revgeocode() uses Google Map API to convert latitude and Longitude into address info, 
  #           the limit (from Google) of 2500 requests per day may limit the running of this code file
  # Reminder:running Line 68 will make over 1200 requests to Google Map API to convert longitude and altitude
  #           to address including zipcode and neighborhood names. It will be a pretty long time to running
  #           (up to 30 min depending on environment).And as mentioned above, Google Map API limits the requests to 2500 per day
# If there are above errors, the code will stop running and raise exception message to instruct more actions


# prepare the workspace
rm(list = ls())
set.seed(2016)

# check if all needed packages are installed, stop running if not
list_of_packages <- c("ggplot2", "ggmap", "car", "caret", "randomForest", "gbm", "lme4", "plyr")
new_packages <- list_of_packages[!(list_of_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  stop("Please make sure packages listed below are installed: ggmap, ggplot2, caret, gbm, rondomForest, lme4, plyr")
}

library(ggplot2)
library(ggmap)
library(car)
library(caret)
library(randomForest)
library(gbm)

# load data and look at the data
if(file.exists("ListingsAndSales.csv")){
  data_orig <- read.csv(file = "ListingsAndSales.csv")
}else{
  stop("Please make sure the data file ListingsAndSales.csv is located in current work directory.")
}

dim(data_orig)
colnames(data_orig)
str(data_orig)
head(data_orig)
summary(data_orig)
pairs(data_orig)

##########################################################
# Data pre-processing
##########################################################

#  1: drop records without SalesDate
data_sold <- data_orig[!(is.na(data_orig$SalesDate)), ]
dim(data_sold)

#  2: calculate days on market of each sold property
data_sold$day_on_market <- as.numeric(as.Date(data_sold$SalesDate) 
                                      - as.Date(data_sold$ListingDate))


#  3: convert Latitude	and Longitude into zip codes and neighborhood names
#prepare only longtitude and latitude info for conversion
gps<-as.data.frame(data_sold[, 14:15])
gps<-gps[c("Longitude", "Latitude")]

gps$Longitude <- round(gps$Longitude/(10^6), digits = 6)
gps$Latitude <- round(gps$Latitude/(10^6), digits = 6)

#acquire zip code and neighborhood info from Google map API based on longitude and Latitude
address <- lapply(1:nrow(gps), function(i) revgeocode(as.numeric(gps[i,1:2]),output = "more"))
if(is.na(address[[length(address)]][[1]]) || is.na(address[[1]][[1]])){
  stop("There might be error in requesting from Google Map API, please check object address and run again.")
}

data_sold$zipcode <-lapply(1:length(address), function(i) as.numeric(levels(address[[i]][["postal_code"]])))
data_sold$neighborhood<-sapply(1:length(address), function(i) toString(address[[i]][["neighborhood"]]))

#drop original longtitude and Latitude info
data_zip_neighbor <- subset(data_sold, select = -c(Latitude, Longitude))
data_zip_neighbor$zipcode <- sapply(data_zip_neighbor$zipcode, factor)
data_zip_neighbor$neighborhood <- sapply(data_zip_neighbor$neighborhood, factor)


#check levels of zipcode and neighborhood
summary(data_zip_neighbor$zipcode)
summary(data_zip_neighbor$neighborhood)

#find records with neighborhood of ""
data_zip_neighbor$zipcode[data_zip_neighbor$neighborhood == ""]
#there are 13 records with neighborhood of "", complete neighborhood based on zipcode
data_zip_neighbor$neighborhood[data_zip_neighbor$zipcode == 98112 
                               & data_zip_neighbor$neighborhood == ""] = "Madison Park"
data_zip_neighbor$neighborhood[data_zip_neighbor$zipcode == 98136
                               & data_zip_neighbor$neighborhood == ""] = "Southwest"
data_zip_neighbor$neighborhood[data_zip_neighbor$zipcode == 98199
                               & data_zip_neighbor$neighborhood == ""] = "Magnolia"


##########################################################
# Calculate means and medians for each segment of market 
# based on zipcode and neighborhood
##########################################################
dom_mean_zip <-aggregate(data_zip_neighbor$day_on_market, 
                    by = list(data_zip_neighbor$zipcode),FUN = mean, na.rm = TRUE)
dom_median_zip <-aggregate(data_zip_neighbor$day_on_market, 
                    by = list(data_zip_neighbor$zipcode),FUN = median, na.rm = TRUE)
dom_mean_zip<- dom_mean_zip[with(dom_mean_zip, order(x)), ]
dom_median_zip<- dom_median_zip[with(dom_median_zip, order(x)), ]

dom_mean_neighborhood <-aggregate(data_zip_neighbor$day_on_market, 
                         by = list(data_zip_neighbor$neighborhood),FUN = mean, na.rm = TRUE)
dom_median_neighborhood <-aggregate(data_zip_neighbor$day_on_market, 
                                  by = list(data_zip_neighbor$neighborhood),FUN = median, na.rm = TRUE)

dom_mean_neighborhood<- dom_mean_neighborhood[with(dom_mean_neighborhood, order(x)), ]
dom_median_neighborhood<- dom_median_neighborhood[with(dom_median_neighborhood, order(x)), ]

# calculate mean and median for all sold properties
dom_mean_total <- mean(data_zip_neighbor$day_on_market)
dom_median_total <- median(data_zip_neighbor$day_on_market)


##########################################################
# Construct a model to estimate how long it will take  
# a listed home to sell
##########################################################
# explore the dependent variable
hist(data_zip_neighbor$day_on_market, main = "Distribution of day_on_market", col = "lavender", xlab = "day_on_market") 
qqnorm(data_zip_neighbor$day_on_market)
qqline(data_zip_neighbor$day_on_market)

# pre-process data for modeling
# drop listing date and sold date, since we have day_on_market
data_reformed <- subset(data_zip_neighbor, select = -c(ListingDate, SalesDate))

# factorize "YearBuilt", "MajorRemodelYear", "SFR"
nms <- c("YearBuilt", "MajorRemodelYear", "SFR")
data_reformed[nms] <- lapply(data_reformed[nms], as.factor) 

# drop RoomTotalCnt and MajorRemodelYear
data_no_na <- subset(data_reformed, select = -c(RoomTotalCnt, MajorRemodelYear))

# look at the data
summary(data_no_na)
# pairwise correlations between variables
pairs(data_no_na)

# quick check depency between preditors and response
# check SFR and day_on_market
dom_mean <-aggregate(data_no_na$day_on_market, 
                     by = list(data_no_na$SFR),FUN = mean, na.rm = TRUE)
dom_median <-aggregate(data_no_na$day_on_market, 
                       by = list(data_no_na$SFR),FUN = median, na.rm = TRUE)

# check factor variables
summary(data_no_na$YearBuilt)
summary(data_no_na$zipcode)
summary(data_no_na$neighborhood)
# merge factor variable levels with similar median of day_on_market
dom_median_year <-aggregate(data_no_na$day_on_market, 
                     by = list(data_no_na$YearBuilt),FUN = median, na.rm = TRUE)
dom_median_zip <-aggregate(data_no_na$day_on_market, 
                          by = list(data_no_na$zipcode),FUN = median, na.rm = TRUE)
dom_median_neighborhood <-aggregate(data_no_na$day_on_market, 
                          by = list(data_no_na$neighborhood),FUN = median, na.rm = TRUE)

dom_median_year <- dom_median_year[order(dom_median_year$x),]
dom_median_zip <- dom_median_zip[order(dom_median_zip$x),]
dom_median_neighbor <- dom_median_neighborhood[order(dom_median_neighborhood$x),]

# construct a hash-table like list: key = a median, value = years with this median of day_on_market
median_year<-levels(as.factor(dom_median_year$x))
list_year<-lapply(1:length(median_year), 
                  function(i) dom_median_year$Group.1[dom_median_year$x == median_year[i]])
names_year<-lapply(1:length(list_year), function(i) toString(list_year[[i]][[1]]))
names(list_year)<-names_year

median_zip<-levels(as.factor(dom_median_zip$x))
list_zip<-lapply(1:length(median_zip), 
                  function(i) dom_median_zip$Group.1[dom_median_zip$x == median_zip[i]])
names_zip<-lapply(1:length(list_zip), function(i) toString(list_zip[[i]][[1]]))
names(list_zip)<-names_zip

median_neighbor<-levels(as.factor(dom_median_neighbor$x))
list_neighbor<-lapply(1:length(median_neighbor), 
                 function(i) dom_median_neighbor$Group.1[dom_median_neighbor$x == median_neighbor[i]])
names_neighbor<-lapply(1:length(list_neighbor), function(i) toString(list_neighbor[[i]][[1]]))
names(list_neighbor)<-names_neighbor


#combine levels of YearBuilt, zip and neighborhood respectively according to median of day_on_market
for(i in 1:length(list_year)){
  data_no_na$YearBuilt[data_no_na$YearBuilt %in% list_year[[i]]] <- names(list_year)[i]
}

for(i in 1:length(list_zip)){
  data_no_na$zipcode[data_no_na$zipcode %in% list_zip[[i]]] <- names(list_zip)[i]
}

for(i in 1:length(list_neighbor)){
  data_no_na$neighborhood[data_no_na$neighborhood %in% list_neighbor[[i]]] <- names(list_neighbor)[i]
}

#combine levels of YearBuilt, zip and neighborhood respectively according to amount
#for neighborhood, combine all levels with <10 records to "other"
small_group_neighbor <- list("Whittier Heights", "South Lake Union", "Ravenna", "Mann", "East Queen Anne",
                             "Interbay", "Rainier View", "Southeast Magnolia", "Broadview",
                             "West Woodland", "View Ridge")
levels(data_no_na$neighborhood) <- c(levels(data_no_na$neighborhood), "other")
data_no_na$neighborhood[data_no_na$neighborhood %in% small_group_neighbor] = "other"
#for YearBuilt, combine all levels with < 10 records to "other"
small_group_year <- list("1903", "1905", "1945", "1955", "1964","1972", "1975", "1980", "1995")
levels(data_no_na$YearBuilt) <- c(levels(data_no_na$YearBuilt), "other")
data_no_na$YearBuilt[data_no_na$YearBuilt %in% small_group_year] = "other"

# drop levels with 0 records
data_no_na$YearBuilt <- factor(data_no_na$YearBuilt)
data_no_na$zipcode <- factor(data_no_na$zipcode)
data_no_na$neighborhood <- factor(data_no_na$neighborhood)

# drop SaleDollarCnt, keep ListPrice or ZestimateDollarCnt
data_no_saleprice <- subset(data_no_na, select = -c(SaleDollarCnt))
data_done_list <- subset(data_no_saleprice, select = -c(ZestimateDollarCnt))
data_done_zestimate <- subset(data_no_saleprice, select = -c(ListPrice))

# drop all records with missing value
data_done_complete<-data_done_zestimate[complete.cases(data_done_zestimate),]

#look at the data before modeling
hist(data_done_complete$day_on_market, main = "Distribution of day_on_market", col = "lavender", xlab = "day_on_market")
qqnorm(data_done_complete$day_on_market)
qqline(data_done_complete$day_on_market)


# modeling and prediction
#   1. training set & testing set separation--0.9 for training, 0.1 for testing
train <- sample(nrow(data_done_complete), round(0.9*nrow(data_done_complete)))
training <- data_done_complete[train,]
testing <- data_done_complete[-train,]

# check predictors after missing values are omitted
#df.tmp <- na.omit(training)
#summary(df.tmp)

# build basic linear regression model
model_lr_1<-lm(day_on_market ~ ., data = training)

# check the model performance
print(summary(model_lr_1))
opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(model_lr_1, las = 1)

outlierTeoparst(model_lr_1)

training_rmse_lr_1<-sqrt(mean((model_lr_1$residuals)^2))
predictions<-predict(model_lr_1, testing)
testing_rmse_lr_1 <- sqrt(mean((testing$day_on_market - predictions)^2))


#   1. apply 10-fold cross-validation to linear regression model
train_control<- trainControl(method = "cv", number = 10, savePredictions = TRUE)
model_cv<- train(day_on_market~., data = training, 
                 trControl = train_control, method="lm", na.action = na.omit)
#check performance of model
print(getTrainPerf(model_cv))
print(summary(model_cv))
predictions<-predict(model_cv, testing)
testing_rmse_cv <- sqrt(mean((testing$day_on_market - predictions)^2))

#use the whole dataset to train
model_cv_all<-train(day_on_market~., data = data_done_complete, 
                    trControl = train_control, method="lm", na.action = na.omit)
print(getTrainPerf(model_cv_all))
print(summary(model_cv_all))

#   2. Random Forest
model_rf <- randomForest(day_on_market ~ ., data = training, 
                         ntree = 100, mtry = 10,
                         importance= T, na.action = na.omit)
print(summary(model_rf))
plot(model_rf)
importance(model_rf)
varImpPlot(model_rf)

training_rmse_rf <- sqrt(mean((training$day_on_market - model_rf$predicted)^2))
predictions<-predict(model_rf, testing)
testing_rmse_rf <- sqrt(mean((testing$day_on_market - predictions)^2))

#use the whole dataset to train
model_rf_all <- randomForest(day_on_market ~ ., data = data_done_complete, 
                         ntree = 100, mtry = 10,
                         importance= T, na.action = na.omit)
print(summary(model_rf_all))
all_rmse_rf <- sqrt(mean((data_done_complete$day_on_market - model_rf_all$predicted)^2))


#   3. Random Forest, try to train only with most important predictors
model_rf_selected <- randomForest(day_on_market ~ YearBuilt 
                                  + neighborhood + zipcode + FinishedSquareFeet
                                  + ZestimateDollarCnt, data = training, 
                         ntree = 100, mtry = 5,
                         importance= T, na.action = na.omit)
print(summary(model_rf_selected))
plot(model_rf_selected)
importance(model_rf_selected)
varImpPlot(model_rf_selected)

training_rmse_rf_selected <- sqrt(mean((training$day_on_market - model_rf_selected$predicted)^2))
predictions<-predict(model_rf_selected, testing)
testing_rmse_rf_selected <- sqrt(mean((testing$day_on_market - predictions)^2))

#use the whole dataset to train
model_rf_selected_all <- randomForest(day_on_market ~ YearBuilt 
                                  + neighborhood + zipcode + FinishedSquareFeet
                                  + ZestimateDollarCnt, data = data_done_complete, 
                                  ntree = 100, mtry = 5,
                                  importance= T, na.action = na.omit)
print(summary(model_rf_selected_all))
all_rmse_rf_selected <- sqrt(mean((data_done_complete$day_on_market - model_rf_selected_all$predicted)^2))

#   4. GBM with 10-fold cross-validation
model_cv_GBM<- train(day_on_market~., data = training, 
                     trControl = train_control, method="gbm", na.action = na.omit, verbose = FALSE)
print(getTrainPerf(model_cv_GBM))
print(summary(model_cv_GBM))
predictions<-predict(model_cv_GBM, testing)
testing_rmse_gbm <- sqrt(mean((testing$day_on_market - predictions)^2))

#use the whole dataset to train
model_cv_GBM_all<- train(day_on_market~., data = data_done_complete, 
                     trControl = train_control, method="gbm", na.action = na.omit, verbose = FALSE)
print(getTrainPerf(model_cv_GBM_all))
print(summary(model_cv_GBM_all))
