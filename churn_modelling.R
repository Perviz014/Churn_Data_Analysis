# Import libraries & dataset ----
library(tidyverse)
library(dplyr)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(scorecard)#Used for woe bin


#1. Import dataset.
df <- read.csv('Churn_Modelling (1).csv')

df %>%  glimpse()

df$Exited <- df$Exited %>% as.factor() 
#2. Remove unneeded columns. 
df <- df[,-1]
#3. Build Churn model. 
iv <- df %>% 
  iv(y = 'Exited') %>% as_tibble() %>%
  mutate(info_value = round(info_value, 3)) %>%
  arrange(desc(info_value))

ivars <- iv %>% 
  filter(info_value>0.02) %>% 
  select(variable) %>% .[[1]] #Excluding not important variables.

df.iv <- df %>% select(Exited,ivars)

df.iv %>% dim()

bins <- df.iv %>% woebin("Exited")

df.woe <- df.iv %>% woebin_ply(bins)

names(df.woe) <- df.woe %>% names() %>% gsub("_woe","",.)


# coef_na
target <- 'Exited'
features <- df.woe %>% select(-Exited) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))

glm <- glm(f, data = df.woe, family = "binomial")#Binomial means for binary classification
glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]#Choosing linearly dependent columns

features <- features[!features %in% coef_na]#Removing them from the dataset

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df.woe, family = "binomial")



while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,"variable"]#Removing first variable with the highest
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df.woe, family = "binomial")
}


glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features 

df.woe <- df.woe %>% select(target,features)

dt_list <- df.woe %>% 
  split_df("Exited", ratio = 0.8, seed = 123)#This time we used split_df instead of h2o.split
#because we are going to work with woe bins

#4. Compare model results for training and test sets. 
train_woe <- dt_list$train
test_woe <- dt_list$test


h2o.init()

train_h2o <- train_woe %>%  as.h2o()
test_h2o <- test_woe %>%  as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
}

model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>% arrange(desc(p_value))

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)#Pulling the coefficients for each variables 

h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)#Visualizing the Influence of each variable for prediction

#5. Evaluate and explain model results using ROC & AUC curves. 

# Prediction & Confision Matrice
pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)#Predicting values

model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')#Finding the threshold by f1 score
eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = dt_list$test$Exited %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")#Roc curve 
#auc-Area under the curve