library(caret)
library(tidyverse)
library(CatEncoders)
library(wrapr)
library(imputeMissings)
library(BBmisc)
library(speedglm)
library(glmnet)

setwd('/Users/sabrinatan/Documents/HomeCredit/all')
app_test <- read.csv('application_test.csv')
app_train <- read.csv('application_train.csv')

ggplot(app_train, aes(TARGET)) +
  geom_histogram(stat = 'count')

plot_missing(app_train)

## Column types
# Number of columns of each type
table(sapply(app_train, class))

# Number of unique entries in each of the factor columns
factorVars <- which(sapply(app_train, is.factor))
sapply(app_train[names(factorVars)], nlevels)

## Encoding: for categorical variables with 2 unique levels label encode for more than 2 one-hot encode
cat <- sapply(app_train, is.factor)
indx_2 <- sapply(app_train[,cat], nlevels) ==2
indx_more <- sapply(app_train[,cat], nlevels) >2
train_2levels <- app_train[,cat][,indx_2]
train_morelevels <- app_train[,cat][,indx_more]

# Label encoding 3 variables - CatEncoders packages
vars3 <- names(which(indx_2))
for(i in vars3) {
  encode <- LabelEncoder.fit(app_train[, i])
  app_train[, i] <- transform(encode, app_train[, i])
}
app_train[,cat][,indx_2]

# One hot encoding remaining - caret package
dmy <- dummyVars("~.", data = app_train)
trsf <- data.frame(predict(dmy, newdata = app_train))
app_train <- trsf

## Alignment of training and testing data 
# Adding factor levels present in train data but not test data
levels(app_test$CODE_GENDER) <- c(levels(app_test$CODE_GENDER), "XNA")
levels(app_test$NAME_INCOME_TYPE) <- c(levels(app_test$NAME_INCOME_TYPE), 'Maternity leave')
levels(app_test$NAME_FAMILY_STATUS) <- c(levels(app_test$NAME_FAMILY_STATUS), 'Unknown')
cattest <- sapply(app_test, is.factor)
indx_2test <- sapply(app_test[ ,cattest], nlevels) ==2
vars3test <- names(which(indx_2test))
for(i in vars3test) {
  encode <- LabelEncoder.fit(app_test[, i])
  app_test[, i] <- transform(encode, app_test[ ,i])
}
app_test[,vars3test]

dmytest <- dummyVars("~.", data = app_test)
trsftest <- data.frame(predict(dmytest, newdata = app_test))
app_test <- trsftest

## Back to EDA
# Outliers
summary(abs(app_train$DAYS_BIRTH)/ 365)
summary(app_train$DAYS_EMPLOYED) # max value should not be positive
plot_histogram(app_train$DAYS_EMPLOYED)

# Subset & examine anomolous clients
anom <- app_train[which(app_train$DAYS_EMPLOYED == 365243),]
not_anom <- setdiff(app_train, anom)
mean(anom$TARGET)
mean(not_anom$TARGET) # anomolies have lower rate of default.

# Create anomolous flag column and fill originals with NaN
app_train <- app_train %>%
  mutate('DAYS_EMPLOYED_ANOM' = (DAYS_EMPLOYED == 365243)) 
app_train$DAYS_EMPLOYED[app_train$DAYS_EMPLOYED == 365243] <- NaN
plot_histogram(app_train$DAYS_EMPLOYED)

# Do same to test data
app_test <- app_test %>%
  mutate('DAYS_EMPLOYED_ANOM' = (DAYS_EMPLOYED == 354243))
app_test$DAYS_EMPLOYED[app_test$DAYS_EMPLOYED == 365243] <- NaN

## Checking correlations between features and target
M <- cor(app_train, use = 'pairwise.complete.obs')
cor_sorted <- as.matrix(sort(M[, 'TARGET'], decreasing = TRUE))
head(cor_sorted)
tail(cor_sorted)

# Looing at age - highest correlation
app_train$DAYS_BIRTH <- abs(app_train$DAYS_BIRTH)
ggplot(app_train, aes((DAYS_BIRTH / 365), fill = as.factor(TARGET))) +
  geom_histogram(bins = 30, color = 'black') +
  theme(legend.position = 'none')
# Kernel density plot
ggplot(app_train, aes((DAYS_BIRTH / 365), color = as.factor(TARGET))) +
  geom_density(stat = 'density', position = 'identity', show.legend = TRUE) +
  labs(title = 'Distribution of Default by Age')

# Try binning the age variable
age_data <- subset(app_train, select = c(TARGET, DAYS_BIRTH))
age_data$YEARS_BIRTH <- age_data$DAYS_BIRTH / 365
age_data <- age_data %>% 
  mutate(YEARS_BINNED = cut(YEARS_BIRTH, breaks = seq(0,115,5)))

age_groups <- age_data %>%
  group_by(YEARS_BINNED) %>%
  summarise(AVG_YEARS_BIRTH = mean(YEARS_BIRTH),
            AVG_DAYS_BIRTH = mean(DAYS_BIRTH),
            AVG_TARGET = mean(TARGET))
age_groups

# plot age bins and average of target
ggplot(age_groups, aes(x = YEARS_BINNED, y = AVG_TARGET)) +
  geom_col() + 
  labs(title = 'Failure to repay by age group')

## Looking at EXT_SOURCE variables - highest negative correlation
ext_data <- subset(app_train, select = c(TARGET, DAYS_BIRTH, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3))
ext_data_cor <- cor(ext_data, use = 'pairwise.complete.obs')
corrplot(ext_data_cor, method = 'square', addCoef.col = 1)

# Look at distribution of EXT_SOURCES by target 
ggplot(ext_data, aes(EXT_SOURCE_1, color = as.factor(TARGET))) +
  geom_density(stat = 'density', position = 'identity', show.legend = TRUE) +
  labs(title = 'Distribution of Default by EXT_SOURCE_1')

ggplot(ext_data, aes(EXT_SOURCE_2, color = as.factor(TARGET))) +
  geom_density(stat = 'density', position = 'identity', show.legend = TRUE) +
  labs(title = 'Distribution of Default by EXT_SOURCE_2')

ggplot(ext_data, aes(EXT_SOURCE_3, color = as.factor(TARGET))) +
  geom_density(stat = 'density', position = 'identity', show.legend = TRUE) +
  labs(title = 'Distribution of Default by EXT_SOURCE_3')

## Feature Engineering
# # Polynomial features
# poly_features <- subset(app_train, select = c(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, DAYS_BIRTH, TARGET))
# poly_features_test <- subset(app_test, select = c(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, DAYS_BIRTH))
# 
# poly_target <- poly_features$TARGET
# poly_features <- poly_features %>%
#   select(-TARGET)
# 
# # Need to impute missing values
# values <- compute(data = poly_features)
# poly_features <- impute(data = poly_features, object = values, method = 'median/mode', flag = FALSE)
# values_test <- compute(data = poly_features_test)
# poly_features_test <- impute(data = poly_features_test, object = values_test, method = 'median/mode', flag = FALSE)
# # 
# # # Create polynomial object
# # poly_transformer <- poly(as.matrix(poly_features) ,degree = 3, raw=TRUE)
# 
# # Train the polynomial features
# lm(formula =  ~ poly(as.matrix(poly_features), degree = 3), data = poly_features)


## Logistic Regression
# Preprocessing: Imputation and Normalization

# Copy of the train data
train <- app_train

# Feature names
features <- names(train)

# Copy of test data
test <- app_test

# Imputing
impute_train <- compute(data = train)
impute_test <- compute(data = test)
train<- impute(data = train, object = impute_train, flag = FALSE)
test<- impute(data = test, object = impute_test, flag = FALSE)

# Normalizing
train <- normalize(train, method = "range", range = c(0, 1))

# Logistic regression
glm_Logit <- glm(TARGET ~., data = train, family = binomial(link = 'logit'), control = list(maxit = 50)) #didn't converge'
glm_test <- glm(TARGET ~ DAYS_BIRTH, data = train, family = binomial(link = 'logit')) #this works
summary(glm_test)

## Trying glmnet 
# for_matrix <- train[, -which(names(train) == 'TARGET')]
# train_matrix <- sparse.model.matrix(~., data = for_matrix)
# target <- train[, 'TARGET']
# 
# fit <- glmnet(train_matrix, target, family = 'binomial')
# plot(fit, xvar = 'dev', label = TRUE)
train_matrix <- sparse.model.matrix(TARGET~., data = train)
target <- train$TARGET

fit <- glmnet(train_matrix, target, family = 'binomial')

plot(fit)
coef(fit, s = 0.001)

# Predict probabilities on test set
test_matrix <- sparse.model.matrix(~., data = test)
pred_test <- predict(fit, newx = test_matrix, type = 'response', s = c(0.01, 0.001, 0))
pred_df <- as.data.frame(pred_test)

submit <- as.data.frame(app_test$SK_ID_CURR)
submit$TARGET <- pred_df$`3`
colnames(submit) <- c('SK_ID_CURR', 'TARGET')
options(scipen = 999)
str(submit)

write.csv(submit, file = 'log_baseline.csv', row.names = FALSE)
