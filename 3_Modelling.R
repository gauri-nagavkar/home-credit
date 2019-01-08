library(tidyverse)
library(rpart)
library(MLmetrics)
library(neuralnet)
library(Matrix)

setwd('/Users/sabrinatan/Documents/HomeCredit/all')
app_test <- read.csv('application_test.csv')
app_train <- read.csv('application_train.csv')

target_ID <- app_train$SK_ID_CURR
app_train$SK_ID_CURR <- NULL

# Partition the train set into train and test
train_sample <- createDataPartition(app_train$TARGET, times = 1, p = 0.6, list = FALSE)
train <- app_train[train_sample, ]
test <- setdiff(app_train, train)

# Save test set actual TARGET and remove
test_TARGET <- test$TARGET
test$TARGET <- NULL

# Imputing
impute_train <- imputeMissings::compute(data = train)
train <- imputeMissings::impute(data = train, object = impute_train, flag = FALSE)
impute_test <- imputeMissings::compute(data = test)
test <- imputeMissings::impute(data = test, object = impute_test, flag = FALSE)

# Adding a missing level to test 
levels(test$NAME_FAMILY_STATUS) <- c(levels(test$NAME_FAMILY_STATUS), 'Unknown')

# Sparse model matrix
train_matrix <- sparse.model.matrix(TARGET~., data = train)
target <- train$TARGET
fit <- glmnet(train_matrix, target, family = 'binomial')

plot(fit)
coef(fit, s = 0)

# Predict probabilities on test set
test_matrix <- sparse.model.matrix(~., data = test)
pred_test <- predict(fit, newx = test_matrix, type = 'response', s = 0)
pred_df <- as.data.frame(pred_test)

# Check accuracy (F1 Score) - 0.9581512
predictions <- pred_df %>%
  mutate('PREDICTION' = ifelse(`1` > 0.5, 1, 0))
F1_Score(test_TARGET, predictions$PREDICTION)

## Training on entire train set --------------------------------------------
# Remove and store test set ID's
test_ID <- app_test$SK_ID_CURR
app_test$SK_ID_CURR <- NULL

# Imputing
impute_app_train <- imputeMissings::compute(data = app_train)
impute_app_test <- imputeMissings::compute(data = app_test)
app_train_imputed <- imputeMissings::impute(data = app_train, object = impute_app_train, flag = FALSE)
app_test_imputed <- imputeMissings::impute(data = app_test, object = impute_app_test, flag = FALSE)

# Insert missing levels
levels(app_test_imputed$NAME_FAMILY_STATUS) <- c(levels(app_test_imputed$NAME_FAMILY_STATUS), 'Unknown')
levels(app_test_imputed$CODE_GENDER) <- c(levels(app_test_imputed$CODE_GENDER), 'XNA')
levels(app_test_imputed$NAME_INCOME_TYPE) <- c(levels(app_test_imputed$NAME_INCOME_TYPE), 'Maternity leave')

# Sparse model matrices
app_train_matrix <- sparse.model.matrix(TARGET ~., data = app_train_imputed)
app_test_matrix <- sparse.model.matrix(~., data = app_test_imputed)
dim(app_train_matrix)
dim(app_test_matrix)

# Logistic regression
target <- app_train_imputed$TARGET
log_fit <- glmnet(app_train_matrix, target, family = 'binomial')
plot(fit)
coef(fit, s = 0)

# Predict probabilities on test set
predicted_test <- predict(fit, newx = app_test_matrix, type = 'response', s = 0)
predicted_df <- as.data.frame(predicted_test)

# Check kaggle submission accuracy - 66.9%
submission2 <- as.data.frame(test_ID)
submission2$TARGET <- predicted_df$`1`
options(scipen = 999)
colnames(submission2) <- c('SK_ID_CURR', 'TARGET')

str(submission2)
write.csv(submission2, file = 'log_baseline2.csv', row.names = FALSE)

## Back to train and test split -----------------------------------------------
# # Preprocessing - normalize
# train <- normalize(train, method = "range", range = c(0, 1))
# test <- normalize(test, method = 'range', range = c(0,1))
# 
# # Sparse model matrices
# train_matrix <- model.matrix(~., data = train)
# test_matrix <- model.matrix(~., data = test)
# dim(train_matrix)
# dim(test_matrix)
# 
# # Fixing colnames 
# colnames(train_matrix) <- make.names(colnames(train_matrix), unique = TRUE)
# colnames(test_matrix) <- make.names(colnames(test_matrix), unique = TRUE)

# NN - too slow
# train_names <- names(as.data.frame(train_matrix))
# mdf <- as.data.frame(train_matrix)
# nn_formula <- paste0('TARGET ~ ', paste(names(mdf[!names(mdf) %in% 'TARGET']), collapse = ' + '))
# nn <- neuralnet(nn_formula, data = train_matrix, hidden = 5, linear.output = FALSE)

## LightGBM 
library(lightgbm)

# Preprocessing: using the train/test split - run up to line 32
train_TARGET <- train$TARGET
train$TARGET <- NULL
train <- normalize(train, method = "range", range = c(0, 1))
test <- normalize(test, method = 'range', range = c(0,1))

# Sparse model matrices
train_matrix <- model.matrix(~., data = train)
test_matrix <- model.matrix(~., data = test)
dim(train_matrix)
dim(test_matrix)

# Fixing colnames
colnames(train_matrix) <- make.names(colnames(train_matrix), unique = TRUE)
colnames(test_matrix) <- make.names(colnames(test_matrix), unique = TRUE)

## Modelling
lgb.train <- lgb.Dataset(train_matrix, label = train_TARGET)
lgb.valid <- lgb.Dataset(test_matrix, label = test_TARGET)

params.lgb = list(objective = "binary", 
                  metric = "auc", 
                  min_data_in_leaf = 1, 
                  min_sum_hessian_in_leaf = 100, 
                  feature_fraction = 1, 
                  bagging_fraction = 1, 
                  bagging_freq = 0)

lgb.model <- lgb.train(params = params.lgb,
                       data = lgb.train,
                       valids = list(val = lgb.valid),
                       learning_rate = 0.06,
                       num_leaves = 7,
                       num_threads = 2,
                       nrounds = 3000,
                       early_stopping_rounds = 200,
                       eval_freq = 50)

# Predictions & submit (run 50-65)
submit_test <- normalize(app_test_imputed, method = 'range', range = c(0,1))
submit_test_matrix <- sparse.model.matrix(~., data = submit_test)

lgb_pred <- predict(lgb.model, data = submit_test_matrix, n = lgb.model$best_iter)
result <- data.frame(SK_ID_CURR = test_ID, TARGET = lgb_pred) 
# write.csv(result, 'lgb_model.csv', row.names = FALSE)
write.csv(result, 'lgb_model3.csv', row.names = FALSE)

## Feature importance - trying to address overfitting
lgbimp <- lgb.importance(lgb.model)
lgb.plot.importance(lgbimp, top_n = 20)

## Feature engineering -------------------------------------------------------------
# Inserting some features found on internet to be significant
app_train$TERM <- app_train$AMT_CREDIT / app_train$AMT_ANNUITY
app_test$TERM <- app_test$AMT_CREDIT / app_test$AMT_ANNUITY


## Error checking - learning curves (run 50 - 65) ----------------------------------
library(mlr)

# Preprocessing
train <- normalize(app_train_imputed, method = "range", range = c(0, 1))
test <- normalize(app_test_imputed, method = 'range', range = c(0,1))
train$TARGET <- as.factor(train$TARGET)

# Task type creation
classif.task <- makeClassifTask(id = 'homecred', data = train, target = 'TARGET', positive = 1)
classif.task

# Learner creation - learning rate increased from 0.05 to 0.06 to address overfitting
classif.lrn <- makeLearner('classif.lightgbm', predict.type = 'prob', fix.factors.prediction = TRUE,
                           metric = "auc", 
                           # min_data_in_leaf = 1, 
                           min_sum_hessian_in_leaf = 100, 
                           feature_fraction = 1, 
                           bagging_fraction = 1, 
                           bagging_freq = 0,
                           learning_rate = 0.06,
                           num_leaves = 7,
                           num_threads = 2,
                           nrounds = 1000,
                           # early_stopping_rounds = 200,
                           eval_freq = 50)
classif.lrn

# Train the learner 
mod <- train(classif.lrn, classif.task)

learningCurve <- generateLearningCurveData(learners = classif.lrn,
                                           task = classif.task,
                                           percs = seq(0.1, 1, by = 0.2),
                                           measures = list(auc, setAggregation(auc, train.mean)),
                                           # measures = list(tp, fp, tn, fn),
                                           resampling = makeResampleDesc(method = "CV", iters = 3, predict ='both'),
                                           show.info = TRUE)

plotLearningCurve(learningCurve, facet = 'learner') +
  labs(title = 'Learning Curve for LightGBM Model')

# Try making predictions & submit
pred <- predict(mod, newdata = test)
pred <- as.data.frame(pred)
result <- data.frame(SK_ID_CURR = test_ID, TARGET = pred$prob.1) 

write.csv(result, 'lgb_model2.csv', row.names = FALSE)

