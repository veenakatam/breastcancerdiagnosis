
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(pROC)
library(MASS)
library(caTools)
library(caret)
library(caretEnsemble)
library(doMC)

##loading data
data <-read.csv("E:/stat final project/breast-cancer-wisconsin-prognostic-data-set/data.csv")
str(data)
##
data$diagnosis <- as.factor(data$diagnosis)
# the 33 column is not right
data[,33] <- NULL
#
data[,33] <- NULL
summary(data)
#
corr_mat <- cor(data[,3:ncol(data)])
corrplot(corr_mat)
##
set.seed(1234)
data_index <- createDataPartition(data$diagnosis, p=0.7, list = FALSE)
train_data <- data[data_index, -1]
test_data <- data[-data_index, -1]
###
pca_res <- prcomp(data[,3:ncol(data)], center = TRUE, scale = TRUE)
plot(pca_res, type="l")
summary(pca_res)

##
pca_df <- as.data.frame(pca_res$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=data$diagnosis)) + geom_point(alpha=0.5)

##
g_pc1 <- ggplot(pca_df, aes(x=PC1, fill=data$diagnosis)) + geom_density(alpha=0.25)  
g_pc2 <- ggplot(pca_df, aes(x=PC2, fill=data$diagnosis)) + geom_density(alpha=0.25)  
grid.arrange(g_pc1, g_pc2, ncol=2)
##
lda_res <- lda(diagnosis~., data, center = TRUE, scale = TRUE) 
lda_df <- predict(lda_res, data)$x %>% as.data.frame() %>% cbind(diagnosis=data$diagnosis)

#

ggplot(lda_df, aes(x=LD1, y=0, col=diagnosis)) + geom_point(alpha=0.5)
ggplot(lda_df, aes(x=LD1, fill=diagnosis)) + geom_density(alpha=0.5)
#
train_data_lda <- lda_df[data_index,]
test_data_lda <- lda_df[-data_index,]
##
fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)
#
model_lda <- train(diagnosis~.,
                   train_data_lda,
                   method="lda2",
                   #tuneLength = 10,
                   metric="ROC",
                   preProc = c("center", "scale"),
                   trControl=fitControl)
#
pred_lda <- predict(model_lda, test_data_lda)
cm_lda <- confusionMatrix(pred_lda, test_data_lda$diagnosis, positive = "M")
cm_lda
#
pred_prob_lda <- predict(model_lda, test_data_lda, type="prob")
#roc_lda <- roc(test_data_lda$diagnosis, pred_prob_lda$M)
#plot(roc_lda)
colAUC(pred_prob_lda, test_data_lda$diagnosis, plotROC=TRUE)
#randomforest
model_rf <- train(diagnosis~.,
                  train_data,
                  method="ranger",
                  metric="ROC",
                  #tuneLength=10,
                  #tuneGrid = expand.grid(mtry = c(2, 3, 6)),
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)
#
pred_rf <- predict(model_rf, test_data)
cm_rf <- confusionMatrix(pred_rf, test_data$diagnosis, positive = "M")
cm_rf
#
model_pca_rf <- train(diagnosis~.,
                      train_data,
                      method="ranger",
                      metric="ROC",
                      #tuneLength=10,
                      #tuneGrid = expand.grid(mtry = c(2, 3, 6)),
                      preProcess = c('center', 'scale', 'pca'),
                      trControl=fitControl)
#
pred_pca_rf <- predict(model_pca_rf, test_data)
cm_pca_rf <- confusionMatrix(pred_pca_rf, test_data$diagnosis, positive = "M")
cm_pca_rf
#
#### KNN
##{r, message=FALSE, warning=FALSE, cache=FALSE}

model_knn <- train(diagnosis~.,
train_data,
method="knn",
metric="ROC",
preProcess = c('center', 'scale'),
tuneLength=10,
trControl=fitControl)



pred_knn <- predict(model_knn, test_data)
cm_knn <- confusionMatrix(pred_knn, test_data$diagnosis, positive = "M")
cm_knn
##
pred_prob_knn <- predict(model_knn, test_data, type="prob")
roc_knn <- roc(test_data$diagnosis, pred_prob_knn$M)
plot(roc_knn)
#

#```{r model_nnet, message=FALSE, warning=FALSE, cache=FALSE}
model_nnet <- train(diagnosis~.,
                    train_data,
                    method="nnet",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    tuneLength=10,
                    trControl=fitControl)
##r script
```{r}
pred_nnet <- predict(model_nnet, test_data)
cm_nnet <- confusionMatrix(pred_nnet, test_data$diagnosis, positive = "M")
cm_nnet
```



#### Neural Networks (NNET) with PCA


#```{r model_pca_nnet, message=FALSE, warning=FALSE, cache=FALSE}
model_pca_nnet <- train(diagnosis~.,
                        train_data,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
```

```{r}
pred_pca_nnet <- predict(model_pca_nnet, test_data)
cm_pca_nnet <- confusionMatrix(pred_pca_nnet, test_data$diagnosis, positive = "M")
cm_pca_nnet
```

#### Neural Networks (NNET) with LDA


```{r model_lda_nnet, message=FALSE, warning=FALSE, cache=FALSE}
model_lda_nnet <- train(diagnosis~.,
                        train_data_lda,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
```

```{r}
pred_lda_nnet <- predict(model_lda_nnet, test_data_lda)
cm_lda_nnet <- confusionMatrix(pred_lda_nnet, test_data_lda$diagnosis, positive = "M")
cm_lda_nnet
``` 


#### SVM with radial kernel


```{r model_svm, message=FALSE, warning=FALSE, cache=FALSE}
model_svm <- train(diagnosis~.,
                   train_data,
                   method="svmRadial",
                   metric="ROC",
                   preProcess=c('center', 'scale'),
                   trace=FALSE,
                   trControl=fitControl)
```

```{r}
pred_svm <- predict(model_svm, test_data)
cm_svm <- confusionMatrix(pred_svm, test_data$diagnosis, positive = "M")
cm_svm
``` 

#### Naive Bayes

```{r model_nb, message=FALSE, warning=FALSE, cache=FALSE}
model_nb <- train(diagnosis~.,
                  train_data,
                  method="nb",
                  metric="ROC",
                  preProcess=c('center', 'scale'),
                  trace=FALSE,
                  trControl=fitControl)
```

```{r}
pred_nb <- predict(model_nb, test_data)
cm_nb <- confusionMatrix(pred_nb, test_data$diagnosis, positive = "M")
cm_nb
``` 


#### Naive Bayes (LDA)

```{r model_lda_nb, message=FALSE, warning=FALSE, cache=FALSE}
model_lda_nb <- train(diagnosis~.,
                      train_data_lda,
                      method="nb",
                      metric="ROC",
                      preProcess=c('center', 'scale'),
                      trace=FALSE,
                      trControl=fitControl)
```

```{r}
pred_lda_nb <- predict(model_lda_nb, test_data_lda)
cm_lda_nb <- confusionMatrix(pred_lda_nb, test_data$diagnosis, positive = "M")
cm_lda_nb
``` 

### Model result comparasion 

Let's compare the models and check their correlation:

```{r}
model_list <- list(RF=model_rf, PCA_RF=model_pca_rf, 
NNET=model_nnet, PCA_NNET=model_pca_nnet, LDA_NNET=model_lda_nnet, 
KNN = model_knn, SVM=model_svm, NB=model_nb, LDA_NB=model_lda_nb)
resamples <- resamples(model_list)

```

#### Correlation between models {.tabset .tabset-fade}

```{r}
model_cor <- modelCor(resamples)
```

##### Plot 
```{r}
corrplot(model_cor)
```

##### Data
```{r}
model_cor
```


#### Comparasion




```{r}
bwplot(resamples, metric="ROC")
```

We see here that some models have a great variability depending of the processed sample (NB). The models LDA_NNET, and LDA_NB achieve a great auc with some variability.

The ROC metric measure the auc of the roc curve of each model. This metric is independent of any threshold.
Let's remember how these models result with the testing dataset. Prediction classes are obtained by default with 
a threshold of 0.5 which could not be the best with an unbalanced dataset like this.


```{r}
cm_list <- list(RF=cm_rf, PCA_RF=cm_pca_rf, 
                NNET=cm_nnet, PCA_NNET=cm_pca_nnet, LDA_NNET=cm_lda_nnet, 
                KNN = cm_knn, SVM=cm_svm, NB=cm_nb, LDA_NB=cm_lda_nb)

```

```{r}
cm_list_results <- sapply(cm_list, function(x) x$byClass)
cm_list_results
```

```{r}
cm_results_max <- apply(cm_list_results, 1, which.is.max)
```

```{r}
output_report <- data.frame(metric=names(cm_results_max), 
                            best_model=colnames(cm_list_results)[cm_results_max],
                            value=mapply(function(x,y) {cm_list_results[x,y]}, 
                                         names(cm_results_max), 
                                         cm_results_max))
rownames(output_report) <- NULL
output_report
```

The best results for sensitivity (detection of breast cases) is LDA_NNET which also has a great F1 score.



## Conclusions

##We have found a model based on neural network and LDA preprocessed data with good results over the test set. 
This model has a **sensibility of 0.984** with a **F1 score of 0.984**  
  
  We have tried an stacked model with a little improvement.

Next things to try:
  
  - use unbalanced techinques (oversampling, SMOTE...) previously to apply the models

- modify models to use a different metric rather than ROC (auc) which takes in consideration the best threshold

- Try different stacking models







