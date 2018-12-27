## home_credit project - Exploratory data analysis

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
