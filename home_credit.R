library(tidyverse)
library(corrplot)
library(DataExplorer)
library(mice)

## home_credit project - classification problem with binary output

setwd('/Users/sabrinatan/Documents/home_credit data/all')

application_test <- read.csv('application_test.csv')
application_train <- read.csv('application_train.csv')
bureau_balance <- read.csv('bureau_balance.csv')
bureau <- read.csv('bureau.csv')
credit_card_balance <- read.csv('credit_card_balance.csv')
HomeCredit_columns <- read.csv('HomeCredit_columns_description.csv')
installments_payments <- read.csv('installments_payments.csv')
POS_CASH_balance <- read.csv('POS_CASH_balance.csv')
previous_applcations <- read.csv('previous_application.csv')
sample_submissison <- read.csv('sample_submission.csv')

summary(application_test)

## Remove ID variables but store them
test_id <- application_test$SK_ID_CURR
train_id <- application_train$SK_ID_CURR
application_test$SK_ID_CURR <- NULL
application_train$SK_ID_CURR <- NULL

## Combine dfs
application_test$TARGET <- NA
all <- rbind(application_train, application_test)

## Tally the target variable - 282,686 0's and 24,825 1's
table(all$TARGET)

## Checking numeric predictors - this did not help
numericVars <- which(sapply(all, is.numeric))
numericVarNames <- names(numericVars)
numVars <- all[, numericVars]
M <- cor(numVars, use = 'pairwise.complete.obs')

#select only high corelations
cor_sorted <- as.matrix(sort(M[,'TARGET'], decreasing = TRUE))
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.05)))
M <- M[CorHigh, CorHigh]

corrplot(M, method = 'circle')

## No features are strongly corr with TARGET. 
## Features with somewhat corr: 1) DAYS_BIRTH, 2) REGION_RATING_CLIENT_W_CITY, 3) DAYS_LAST_PHONE_CHANGE, 4) DAYS_ID_PUBLISH, 5) REG_CITY_NOT_WORK_CITY

## 1) DAYS_BIRTH
## Convert Days Birth to age
all <- all %>%
  mutate(AGE = as.factor(round(abs(DAYS_BIRTH) / 365)))
all$DAYS_BIRTH <- NULL

## Plotting age vs default - likely bin this variable later
l <- ggplot(all[!(is.na(all$TARGET)),], aes(AGE, fill = TARGET)) +
  geom_histogram(stat = 'count', binwidth = 100)
l

## 2) REGION_RATING_CLIENT_W_CITY - Our rating of the region where client lives with taking city into account (1,2,3)
cor(all$REGION_RATING_CLIENT, all$REGION_RATING_CLIENT_W_CITY)
# Plotting rating vs. default
w <- ggplot(all[!(is.na(all$TARGET)),], aes(REGION_RATING_CLIENT_W_CITY, fill = TARGET)) +
  geom_bar(stat = 'count')
w
# Note that row 17178 has invalid entry (-1)

## 3) DAYS_LAST_PHONE_CHANGE - How many days before application did client change phone
# Convert to age
all <- all %>%
  mutate(AGE_LAST_PHONE_CHANGE = as.factor(round(abs(DAYS_LAST_PHONE_CHANGE) / 365)))
all$DAYS_LAST_PHONE_CHANGE <- NULL

p <- ggplot(all[!(is.na(all$TARGET)),], aes(AGE_LAST_PHONE_CHANGE, fill = TARGET)) +
  geom_histogram(stat = 'count', binwidth = 10)
p

## 4) DAYS_ID_PUBLISH - How many days before the application did client change the identity document with which he applied for the loan
## Convert to age 
all <- all %>%
  mutate(AGE_ID_PUBLISH = as.factor(round(abs(DAYS_ID_PUBLISH)/ 365)))
all$DAYS_ID_PUBLISH <- NULL

i <- ggplot(all[!(is.na(all$TARGET)),], aes(AGE_ID_PUBLISH, fill = TARGET)) +
  geom_histogram(stat = 'count', binwidth = 10)
i

## 5) REG_CITY_NOT_WORK_CITY - Flag if client's permanent address does not match work address (1=different, 0=same, at city level)
r <- ggplot(all[!(is.na(all$TARGET)),], aes(REG_CITY_NOT_WORK_CITY, fill = TARGET)) +
  geom_histogram(stat = 'count', binwidth = 100)
r

## Check missing
plot_missing(all[!(is.na(all$TARGET)),])
md.pattern(all[!(is.na(all$TARGET)),])

# Quickly impute missing data with mice package - haven't dealt w colinear variables so pmm doesn't work. using cart - THIS DOESN'T WORK
numVars <- names(all[!(is.na(all$TARGET)),])[sapply(all[!(is.na(all$TARGET)),], is.numeric)]
train <- all[!(is.na(all$TARGET)),]
tempTrain <- mice(data = train[numVars], m = 5, method = 'norm.predict', maxit = 50, seed = 500)

