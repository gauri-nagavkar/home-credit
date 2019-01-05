home\_credit
================
Sabrina Tan
2019-01-05

<br>

<br>

Kaggle: Home Credit Default Risk
================================

#### December 2018 - January 2019

During a holiday break, I decided to learn more about machine learning using [Machine Learning by Stanford University](https://www.coursera.org/learn/machine-learning/home/welcome), and try to solidify the knowledge by applying the concepts learned. I chose Kaggle's [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) problem, being one of the more popular datasets with plentiful resources online in case I got stuck, and one that I could immediately see how concepts from the course could be applied. The problem is a supervised classification problem with binary output. This is a documentation of the progress made thus far. <br>

Exploratory Data Analysis
-------------------------

I looked to [this kernel](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction) to guide me through the initial data exploration prior to modelling and started off using the main application\_train and application\_test datasets.

``` r
## Libraries used:
library(rmarkdown)
library(tidyverse)
library(questionr)
library(caret)
library(corrplot)
library(imputeMissings)
library(glmnet)
library(Matrix)
library(MLmetrics)
library(lightgbm)
library(BBmisc)
library(mlr)
```

``` r
## Importing data:
setwd('/Users/sabrinatan/Documents/HomeCredit/all')
app_test <- read.csv('application_test.csv')
app_train <- read.csv('application_train.csv')
```

<br>

#### The Target Variable

The target variable in this problem is the column 'TARGET', indicating whether the client had difficulties repaying the loan. 1 indicates yes (positive), and 0 indicates no (negative). <br> ![](home_credit_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r
table(app_train$TARGET)
```

    ## 
    ##      0      1 
    ## 282686  24825

Since there are many more 0's than 1's, this is an imbalanced classification problem. <br>

#### Missing values

``` r
## The top 20 features with missing values
freq.na(app_train)[1:20,]
```

    ##                          missing  %
    ## COMMONAREA_AVG            214865 70
    ## COMMONAREA_MODE           214865 70
    ## COMMONAREA_MEDI           214865 70
    ## NONLIVINGAPARTMENTS_AVG   213514 69
    ## NONLIVINGAPARTMENTS_MODE  213514 69
    ## NONLIVINGAPARTMENTS_MEDI  213514 69
    ## LIVINGAPARTMENTS_AVG      210199 68
    ## LIVINGAPARTMENTS_MODE     210199 68
    ## LIVINGAPARTMENTS_MEDI     210199 68
    ## FLOORSMIN_AVG             208642 68
    ## FLOORSMIN_MODE            208642 68
    ## FLOORSMIN_MEDI            208642 68
    ## YEARS_BUILD_AVG           204488 66
    ## YEARS_BUILD_MODE          204488 66
    ## YEARS_BUILD_MEDI          204488 66
    ## OWN_CAR_AGE               202929 66
    ## LANDAREA_AVG              182590 59
    ## LANDAREA_MODE             182590 59
    ## LANDAREA_MEDI             182590 59
    ## BASEMENTAREA_AVG          179943 59

``` r
## Table of missing values by column
na_count <- data.frame('Column' = colnames(app_train), 'Count' = sapply(app_train, function(x) sum(length(which(is.na(x))))))
rownames(na_count) <- NULL
na_count <- na_count[order(-na_count$Count),]
head(na_count)
```

    ##                      Column  Count
    ## 49           COMMONAREA_AVG 214865
    ## 63          COMMONAREA_MODE 214865
    ## 77          COMMONAREA_MEDI 214865
    ## 57  NONLIVINGAPARTMENTS_AVG 213514
    ## 71 NONLIVINGAPARTMENTS_MODE 213514
    ## 85 NONLIVINGAPARTMENTS_MEDI 213514

<br>

#### Columns

This section explores the types of columns in the Home Credit dataset.

``` r
# Number of columns of each type
table(sapply(app_train, class))
```

    ## 
    ##  factor integer numeric 
    ##      16      41      65

``` r
# Number of unique entries in each of the factor columns
factorVars <- which(sapply(app_train, is.factor))
sapply(app_train[names(factorVars)], nlevels)
```

    ##         NAME_CONTRACT_TYPE                CODE_GENDER 
    ##                          2                          3 
    ##               FLAG_OWN_CAR            FLAG_OWN_REALTY 
    ##                          2                          2 
    ##            NAME_TYPE_SUITE           NAME_INCOME_TYPE 
    ##                          8                          8 
    ##        NAME_EDUCATION_TYPE         NAME_FAMILY_STATUS 
    ##                          5                          6 
    ##          NAME_HOUSING_TYPE            OCCUPATION_TYPE 
    ##                          6                         19 
    ## WEEKDAY_APPR_PROCESS_START          ORGANIZATION_TYPE 
    ##                          7                         58 
    ##         FONDKAPREMONT_MODE             HOUSETYPE_MODE 
    ##                          5                          4 
    ##         WALLSMATERIAL_MODE        EMERGENCYSTATE_MODE 
    ##                          8                          3

<br>

#### Checking correlations between features and target variable

First one-hot encoding the categorical variables using the caret package so the cor function can be run.

``` r
dmy <- dummyVars("~.", data = app_train)
trsf <- data.frame(predict(dmy, newdata = app_train))
app_train <- trsf
```

Most positive and most negative correlations with the target variable:

    ## Warning in cor(app_train, use = "pairwise.complete.obs"): the standard
    ## deviation is zero

    ##                                   [,1]
    ## TARGET                      1.00000000
    ## DAYS_BIRTH                  0.07823931
    ## REGION_RATING_CLIENT_W_CITY 0.06089267
    ## REGION_RATING_CLIENT        0.05889901
    ## NAME_INCOME_TYPE.Working    0.05748118
    ## DAYS_LAST_PHONE_CHANGE      0.05521848

    ##                                             [,1]
    ## NAME_INCOME_TYPE.Pensioner           -0.04620942
    ## CODE_GENDER.F                        -0.05470405
    ## NAME_EDUCATION_TYPE.Higher.education -0.05659264
    ## EXT_SOURCE_1                         -0.15531713
    ## EXT_SOURCE_2                         -0.16047167
    ## EXT_SOURCE_3                         -0.17891870

<br>

#### Age variable: highest positive correlation

The age variable is stored as a negative value (days before application date) in the dataset. They are converted to a true age for exploration purposes. <br> Plots: ![](home_credit_files/figure-markdown_github/unnamed-chunk-10-1.png)![](home_credit_files/figure-markdown_github/unnamed-chunk-10-2.png)

Binning the age variable:

``` r
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
```

    ## # A tibble: 10 x 4
    ##    YEARS_BINNED AVG_YEARS_BIRTH AVG_DAYS_BIRTH AVG_TARGET
    ##    <fct>                  <dbl>          <dbl>      <dbl>
    ##  1 (20,25]                 23.4          8533.     0.123 
    ##  2 (25,30]                 27.8         10155.     0.111 
    ##  3 (30,35]                 32.5         11855.     0.103 
    ##  4 (35,40]                 37.6         13708.     0.0894
    ##  5 (40,45]                 42.5         15498.     0.0785
    ##  6 (45,50]                 47.5         17324.     0.0742
    ##  7 (50,55]                 52.6         19196.     0.0670
    ##  8 (55,60]                 57.5         20984.     0.0553
    ##  9 (60,65]                 62.4         22781.     0.0527
    ## 10 (65,70]                 66.6         24293.     0.0373

Plot of the new age groups and average of the target variable:

![](home_credit_files/figure-markdown_github/unnamed-chunk-12-1.png)

Now we can see a clear trend between default and age. <br>

#### EXT\_SOURCE variables - highest negative correlations

The external source variables (EXT\_SOURCE1, EXT\_SOURCE\_2, EXT\_SOURCE\_3) correspond to the client's "Normalized score from external data source".

Correlation plot:

``` r
ext_data <- subset(app_train, select = c(TARGET, DAYS_BIRTH, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3))
ext_data_cor <- cor(ext_data, use = 'pairwise.complete.obs')
corrplot(ext_data_cor, method = 'circle', tl.col='black', tl.cex=0.6, tl.srt=30,
         addCoef.col = 'black', 
         col = colorRampPalette(c('white','orchid4','orange'))(200)) 
```

![](home_credit_files/figure-markdown_github/unnamed-chunk-13-1.png)

Looking at the distributions of default by EXT\_SOURCE variables:

    ## Warning: Removed 173378 rows containing non-finite values (stat_density).

    ## Warning: Removed 660 rows containing non-finite values (stat_density).

    ## Warning: Removed 60965 rows containing non-finite values (stat_density).

![](home_credit_files/figure-markdown_github/unnamed-chunk-14-1.png)![](home_credit_files/figure-markdown_github/unnamed-chunk-14-2.png)![](home_credit_files/figure-markdown_github/unnamed-chunk-14-3.png) <br> <br>

Modelling
---------

### Logistic regression

The baseline model I used was logistic regression using glmnet. I first modelled using a 60/40 split on the application\_train dataset, then trained on the entire application\_train dataset for the baseline submission to Kaggle. <br>

#### Partitioned data

``` r
setwd('/Users/sabrinatan/Documents/HomeCredit/all')
app_test <- read.csv('application_test.csv')
app_train <- read.csv('application_train.csv')

target_ID <- app_train$SK_ID_CURR
app_train$SK_ID_CURR <- NULL
test_ID <- app_test$SK_ID_CURR
app_test$SK_ID_CURR <- NULL
```

``` r
# Partition the train set into train and test
train_sample <- createDataPartition(app_train$TARGET, times = 1, p = 0.6, list = FALSE)
train <- app_train[train_sample, ]
test <- setdiff(app_train, train)
# Save test set actual TARGET and remove
test_TARGET <- test$TARGET
test$TARGET <- NULL
```

<br>

##### Imputation

Missing values had to be addressed before modelling. Here, I impute missing values with the median.

``` r
impute_train <- imputeMissings::compute(data = train)
train <- imputeMissings::impute(data = train, object = impute_train, flag = FALSE)
impute_test <- imputeMissings::compute(data = test)
test <- imputeMissings::impute(data = test, object = impute_test, flag = FALSE)
```

Errors arose due to the train and test sets having different factor levels, here I inject them into the test set:

``` r
levels(test$NAME_FAMILY_STATUS) <- c(levels(test$NAME_FAMILY_STATUS), 'Unknown')
levels(test$NAME_FAMILY_STATUS) <- c(levels(test$NAME_FAMILY_STATUS), 'Unknown')
levels(test$CODE_GENDER) <- c(levels(test$CODE_GENDER), 'XNA')
levels(test$NAME_INCOME_TYPE) <- c(levels(test$NAME_INCOME_TYPE), 'Maternity leave')
```

<br> I initially used base R's glm function, however this was running extremely slow. I decided to convert to sparse model matrices and use glmnet.

``` r
# Sparse model matrix
train_matrix <- sparse.model.matrix(TARGET~., data = train)
target <- train$TARGET
fit <- glmnet(train_matrix, target, family = 'binomial')
```

<br>

Predicting probabilities on the test set:

``` r
test_matrix <- sparse.model.matrix(~., data = test)
pred_test <- predict(fit, newx = test_matrix, type = 'response', s = 0)
pred_df <- as.data.frame(pred_test)
```

<br>

Checking accuracy on the test set using F1 score:

``` r
predictions <- pred_df %>%
  mutate('PREDICTION' = ifelse(`1` > 0.5, 1, 0))

F1_Score(test_TARGET, predictions$PREDICTION)
```

    ## [1] 0.9581155

This is the F1 score achieved using logistic regression. <br>

#### Kaggle submission 1

``` r
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
```

<br>

The initial score returned by Kaggle was 0.66825.

<br> <br>

### LightGBM

The next model I decided to try is LightGBM due to its popularity and success on Kaggle (despite trees not being covered in Stanford's ML course). I used [this kernel](https://www.kaggle.com/couyang/home-credit-eda-lightgbm) as a guide.

<br> Here I am using the 60/40 partitioned data from above since I need to enter "valids" as a parameter in lightgbm. Variable names are fixed (spaces replaced with '.'). <br>

``` r
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

lgb.model <- lightgbm::lgb.train(params = params.lgb,
                       data = lgb.train,
                       valids = list(val = lgb.valid),
                       learning_rate = 0.06,
                       num_leaves = 7,
                       num_threads = 2,
                       nrounds = 3000,
                       early_stopping_rounds = 200,
                       eval_freq = 50)
```

#### Kaggle submission 2

``` r
submit_test <- normalize(app_test_imputed, method = 'range', range = c(0,1))
submit_test_matrix <- sparse.model.matrix(~., data = submit_test)

lgb_pred <- predict(lgb.model, data = submit_test_matrix, n = lgb.model$best_iter)
result <- data.frame(SK_ID_CURR = test_ID, TARGET = lgb_pred) 
# write.csv(result, 'lgb_model.csv', row.names = FALSE)
```

<br>

Using LightGBM, the score given by Kaggle is 0.73300. <br> <br>

Error Analysis
--------------

Andrew Ng outlines in his course that the step to follow the baseline model implementation is error analysis. Learning curves are discussed as a tool to identify cause of error. This plot will be created, but first, a feature importance graph: <br>

``` r
lgbimp <- lgb.importance(lgb.model)
lgb.plot.importance(lgbimp, top_n = 20)
```

![](home_credit_files/figure-markdown_github/unnamed-chunk-27-1.png)

<br>

#### Error Curve Analysis

I used the mlr package to generate a learning curve for the LightGBM model.

<br>

I used a custom classification LightGBM learner from [here](https://github.com/ja-thomas/autoxgboost/blob/boosting_backends/R/RLearner_classif_lightgbm.R).

<br> Creating the "Task type" and "Learner":

``` r
# Task type creation
classif.task <- makeClassifTask(id = 'homecred', data = train, target = 'TARGET', positive = 1)
# classif.task

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
# classif.lrn
```

<br> Training the learner:

``` r
mod <- train(classif.lrn, classif.task)

learningCurve <- generateLearningCurveData(learners = classif.lrn,
                                           task = classif.task,
                                           percs = seq(0.1, 1, by = 0.2),
                                           measures = list(auc, setAggregation(auc, train.mean)),
                                           # measures = list(tp, fp, tn, fn),
                                           resampling = makeResampleDesc(method = "CV", iters = 3, predict ='both'),
                                           show.info = FALSE)
```

<br> Learning curve plot:

``` r
plotLearningCurve(learningCurve, facet = 'learner') +
  scale_fill_manual(values=c('orange', 'orchid4'))+
  scale_color_manual(values =c('orange','orchid4')) +
  labs(title = 'Learning Curve for LightGBM Model')
```

![](home_credit_files/figure-markdown_github/unnamed-chunk-33-1.png)

<br>

From this error curve plot, I can deduce that the model is producing results with high variance (large gap between curves; overfitting). Going forward, to improve the model, options to address overfitting are:

-   Getting more training examples (not viable as this is a Kaggle competition

-   Less features: I could refer to the EDA portion of the project to identify the most importance features to keep (potentially use feature engineering to generate more important features and drop insignificant ones)
