# Data visualization
# CSV file I/O, e.g. the read_csv function

library(ggplot2) 
library(readr) 
library(caret)
library(dplyr)
library(ROCR)
#input data files available
system("ls ../input")
options(warn = -1)
#loading libraries
bc <- read.csv("E:/stat final project/breast-cancer-wisconsin-prognostic-data-set/data.csv", header = T)


#Create dummy variable dummy.outcome
bc$dummy.outcome <- ifelse(bc$diagnosis == 'M', 1, 0)
bc$dummy.outcome <- as.factor(bc$dummy.outcome)
#remove id, diagnosis and X column
bc <- bc[,c(-1, -2, -33)]
#glimpse(bcancer.data)

#Plot first 10 variables to check correlation
windows(width=9,height=9)
plot(bc[c(1:9,31)], col = bc$dummy.outcome)
#Variable reduction
#Get all the numeric variables
numeric.data <- bc[sapply(bc, is.numeric)]

#Apply cor function
descr.cor <- cor(numeric.data)

#Get all the correlated variables where correlation coefficient >= 0.7
highly.correlated <- findCorrelation(descr.cor, cutoff = 0.7, verbose = T, names = T)
highly.cor.col <- colnames(numeric.data[highly.correlated])


#Remove the correlated variables
bc <- bc[, -which(colnames(bc) %in% highly.cor.col)]
glimpse(bc)
##
#Data does not contain any missing value
#Splitting the data into train(70%) and test(30%)
index <- createDataPartition(bc$dummy.outcome, p=0.7, list = F)
train <- bc[index,]
test <- bc[-index,]


#Evaluate algorithms to check which perform better than other to 
#Fine tune selected algorithm later
#Checking Logistic Regression and Random Forest
#Using caret package

dataset <- bc
#Set the control parameters. We will use 10 fold cross validation 3 times
control <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)
#Random number
seed <- 7
metric <- 'Accuracy'

#logistic regression
set.seed(seed)
fit.glm <- train(dummy.outcome ~ ., data = train, method = 'glm', 
                 metric = metric, trControl = control)

#random forest
set.seed(seed)
fit.rf <- train(dummy.outcome ~ ., data = train, method = 'rf', 
                metric = metric, trControl = control)






results <- resamples(list(logistic = fit.glm, rf = fit.rf))

summary(results)
windows(width=9,height=9)
bwplot(results)
windows(width=9,height=9)
dotplot(results)



#Let's fine tune Logistic Regression first
fit.glm <- glm(formula = dummy.outcome ~ ., data = train, family = binomial(link = 'logit'), 
               control = list(maxit=100))
summary(fit.glm)

#Predict on train data
p <- predict(fit.glm, train[,-11], type = 'response')
p <- ifelse(p > 0.5, 1, 0)

print(paste('Model Accuracy ', mean(p == train$dummy.outcome)))

#ROC
pred.roc <- prediction(p, train$dummy.outcome)
perf <- performance(pred.roc, measure = 'tpr', x.measure = 'fpr')
windows(width=9,height=9)
plot(perf)

#Area under the curve
auc <- performance(pred.roc, measure = 'auc') 
auc <- unlist(slot(auc, 'y.values'))
print(paste('AUC ', auc))

#Removing symmetry_mean, symmetry_worst and checking the results to fine tune regression
glm.fit2 <- glm(formula = dummy.outcome ~ texture_mean + area_mean +  
                  texture_se + smoothness_se + symmetry_se + fractal_dimension_se + 
                  smoothness_worst + fractal_dimension_worst, 
                data = train, family = binomial(link = 'logit'), 
                control = list(maxit=100))
summary(glm.fit2)

#Predict on train data
p <- predict(glm.fit2, train[,-11], type = 'response')
p <- ifelse(p > 0.5, 1, 0)

print(paste('Model Accuracy on Train', mean(p == train$dummy.outcome)))

#ROC
pred.roc <- prediction(p, train$dummy.outcome)
perf <- performance(pred.roc, measure = 'tpr', x.measure = 'fpr')
windows(width=9,height=9)
plot(perf)

#Area under the curve
auc <- performance(pred.roc, measure = 'auc') 
auc <- unlist(slot(auc, 'y.values'))
print(paste('AUC on Train', auc))


#Fine tuning Random Forest
randomForest.fit <- randomForest(dummy.outcome ~ ., data = bc, 
                       importance = TRUE, ntree = 1000)
randomForest.fit
windows(width=9,height=9)
varImpPlot(randomForest.fit)

#Remove texture_se and check accurracy
randomForest.fit2 <- randomForest(dummy.outcome ~ texture_mean + area_mean + symmetry_mean + 
                          smoothness_se + symmetry_se + fractal_dimension_se + smoothness_worst + 
                          symmetry_worst + fractal_dimension_worst, data = bc, 
                        importance = TRUE, ntree = 1000)
randomForest.fit2
windows(width=9,height=9)
varImpPlot(randomForest.fit2)
#Similarly we can check error rate by removing different variables

