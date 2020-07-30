#libraries need to breast cancer 
any(is.na(bc))li
missmap(bc, main='Missing data', col=c("yellow","black"), legend = FALSE)
bc$X <- NULL
bc$ids <- bc$id
bc$id <- NULL
bc$diagnosis <- as.factor(bc$diagnosis)
str(bc)

##correlation
bc.cor <- test <- select(bc,-ids)
bc.cor$diagnosis <- as.numeric(bc$diagnosis)
B <- cor(bc.cor)
corrplot(B, method="circle")
##R script for breast cancer prognostic and diagnostic analysis
library(randomForest)
##viewing database
view(bc)
head(bc)
tail(bc)
str(bc)
##coding
any(is.na(bc))
missmap(bc, main='Missing data', col=c("yellow","black"), legend = FALSE)
##
bc$X <- NULL
bc$ids <- bc$id
bc$id <- NULL
bc$diagnosis <- as.factor(bc$diagnosis)
str(bc)

##


## c.<a id='id4'>Check out correlation between data</a>



bc.cor <- test <- select(bc,-ids)
bc.cor$diagnosis <- as.numeric(bc$diagnosis)
B <- cor(bc.cor)
corrplot(B, method="circle")
##
windows(width=9,height=9)
ggplot(bc,aes(perimeter_mean, area_mean)) + 
  geom_point(aes(color=factor(diagnosis)),alpha=0.5) 
+ scale_fill_discrete(name="diagnosis", breaks=c("0", "1"), 
                      labels=c("M", "B")) + 
  labs(title = "diagnosis based on perimeter and area mean")
ggplot(bc,aes(symmetry_mean, smoothness_se)) +
  geom_point(aes(color=factor(diagnosis)),alpha=0.5) +
  scale_fill_discrete(name="diagnosis", breaks=c("0", "1"), 
                      labels=c("M", "B")) + 
  labs(title = "diagnosis based on symetry and smoothness")

##split data
split <- sample.split(bc$diagnosis, Split = 0.7)
train <- subset(bc, split == T)
test <- subset(bc, split == F)
#train data
log.model <- glm(formula=diagnosis ~ . , family = binomial(link='logit'),data = train)
summary(log.model)
fitted.probabilities <- predict(log.model,newdata=test[2:32],type='response')

c <-table(test$diagnosis, fitted.probabilities > 0.5)
c
c.t <-sum(diag(c))/sum(c)

print(c.t)
##
rf.model <- randomForest(diagnosis ~., data = train)

#prediction accuracy
predicted.values <- predict(rf.model, test[2:32])
d <- table(predicted.values, test$diagnosis)
print(d)
d.t <-sum(diag(d))/sum(d)
print(d.t)

#train model
rf.model <- randomForest(diagnosis ~., data = train)

#prediction accuracy
predicted.values <- predict(rf.model, test[2:32])
d <- table(predicted.values, test$diagnosis)
print(d)
d.t <-sum(diag(d))/sum(d)
print(d.t)

###
#normalize data
normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

bc1<- as.data.frame(lapply(bc[2:30],normalize))
bc1$diagnosis <- bc$diagnosis

#split data
split <- sample.split(bc1$diagnosis, Split = 0.7)
train <- subset(bc1, split == T)
test <- subset(bc1, split == F)

#predict with k=1
predicted.bc <- knn(train[1:29],test[1:29],train$diagnosis,k=1)

#missclassification rate
mean(test$diagnosis != predicted.bc)

# create k values fuction
predicted.bc <-NULL
error.rate <-NULL

for(i in 1:10){
  predicted.bc <- predicted.species <- knn(train[1:29],test[1:29],train$diagnosis,k=i)
  error.rate[i] <- mean(test$diagnosis != predicted.bc)
}

k.values <- 1:10
error.df <- data.frame(error.rate,k.values)

#plot elbow
pl <- ggplot(error.df,aes(x=k.values,y=error.rate)) + geom_point()
pl + geom_line(lty="dotted",color='red')

#choose k
predicted.bc <- knn(train[1:29],test[1:29],train$diagnosis,k=5)
mean(test$diagnosis != predicted.bc)

#confusion matrix
e <-table(test$diagnosis, predicted.bc)
print(e)
e.t <-sum(diag(e))/sum(e)
print(e.t)

model <- svm(diagnosis ~., data = train)
summary(model)

##
#train model
model <- svm(diagnosis ~., data = train)
summary(model)

#predict
pred.values <- predict(model, test[1:29])
table(pred.values, test$diagnosis)

#tune model
tune.results <-tune(svm, train.x = train[1:29], train.y = train$diagnosis, kernal = 'radial', ranges=list(cost = c(0.1,1,10), gamma = c(0.5, 1, 2)))
summary(tune.results)

#tuned svm
tuned.svm <-svm(diagnosis~., data = train, kernal='radial', cost =10, gamma=0.5)
summary(tuned.svm)

#predict again
pred.values <- predict(tuned.svm, test[1:29])
f <- table(pred.values, test$diagnosis)
f.t <-sum(diag(f))/sum(f)
print(f.t)

pred.values <- predict(tuned.svm, test[1:29])
f <- table(pred.values, test$diagnosis)
f.t <-sum(diag(f))/sum(f)
print(f.t)

##
#normalize data
normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

bc1<- as.data.frame(lapply(bc[2:31],normalize))
bc1$diagnosis <- bc$diagnosis
bc1$diagnosis <- as.numeric(bc1$diagnosis)

binary <- function(dg){
  for(i in 1:length(dg)){
    if(dg[i] == 1){
      dg[i] <- 0
    }else{
      dg <- 1
    }
  }
  return(dg)
}

bc1$diagnosis <-sapply(bc1$diagnosis,binary)

#set up split
split <- sample.split(bc1$diagnosis, Split = 0.7)
train <- subset(bc1, split == T)
test <- subset(bc1, split == F)

#train model
nn <- neuralnet(diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + 
                  smoothness_mean + compactness_mean + concavity_mean + concave.points_mean + 
                  symmetry_mean + fractal_dimension_mean + radius_se + texture_se + 
                  perimeter_se + area_se + smoothness_se + compactness_se + 
                  concavity_se + concave.points_se + symmetry_se + fractal_dimension_se + 
                  radius_worst + texture_worst + perimeter_worst + area_worst + 
                  smoothness_worst + compactness_worst + concavity_worst + 
                  concave.points_worst + symmetry_worst + fractal_dimension_worst, data=train, hidden = c(5,3), linear.output = FALSE)

#predict
predicted.nn.values <- compute(nn, test[,1:30])

#round predicted values
predictions <- sapply(predicted.nn.values$net.result,round)

#table
g <-table(predictions, test$diagnosis)
g.t <- sum(diag(g))/sum(g)
print(g.t)


#set k values
k.values <- 1:10
error.df <- data.frame(error.rate,k.values)

#plot elbow
pl <- ggplot(error.df,aes(x=k.values,y=error.rate)) + geom_point()
pl + geom_line(lty="dotted",color='red')

#choose k
predicted.bc <- knn(train[1:29],test[1:29],train$diagnosis,k=5)
mean(test$diagnosis != predicted.bc)

#confusion matrix
e <-table(test$diagnosis, predicted.boobs)
print(e)
e.t <-sum(diag(e))/sum(e)
print(e.t)

###code
#confusion matrix
e <-table(test$diagnosis, predicted.bc)
print(e)
e.t <-sum(diag(e))/sum(e)
print(e.t)


# V. <a id='id8'>Support Vector Machine</a>


#train model
model <- svm(diagnosis ~., data = train)
summary(model)

#predict
pred.values <- predict(model, test[1:29])
table(pred.values, test$diagnosis)

#tune model
tune.results <-tune(svm, train.x = train[1:29], train.y = train$diagnosis, kernal = 'radial', ranges=list(cost = c(0.1,1,10), gamma = c(0.5, 1, 2)))
summary(tune.results)

#tuned svm
tuned.svm <-svm(diagnosis~., data = train, kernal='radial', cost =10, gamma=0.5)
summary(tuned.svm)

#predict again
pred.values <- predict(tuned.svm, test[1:29])
f <- table(pred.values, test$diagnosis)
f.t <-sum(diag(f))/sum(f)
print(f.t)


# VI. <a id='id9'>Neural Nets</a>


#normalize data
normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

bc1<- as.data.frame(lapply(bc[2:31],normalize))
bc1$diagnosis <- bc$diagnosis
bc1$diagnosis <- as.numeric(bc1$diagnosis)

binary <- function(dg){
  for(i in 1:length(dg)){
    if(dg[i] == 1){
      dg[i] <- 0
    }else{
      dg <- 1
    }
  }
  return(dg)
}

bc1$diagnosis <-sapply(bc1$diagnosis,binary)

#set up split
split <- sample.split(bc1$diagnosis, Split = 0.7)
train <- subset(bc1, split == T)
test <- subset(bc1, split == F)

#train model
nn <- neuralnet(diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + 
                  smoothness_mean + compactness_mean + concavity_mean + concave.points_mean + 
                  symmetry_mean + fractal_dimension_mean + radius_se + texture_se + 
                  perimeter_se + area_se + smoothness_se + compactness_se + 
                  concavity_se + concave.points_se + symmetry_se + fractal_dimension_se + 
                  radius_worst + texture_worst + perimeter_worst + area_worst + 
                  smoothness_worst + compactness_worst + concavity_worst + 
                  concave.points_worst + symmetry_worst + fractal_dimension_worst, data=train, hidden = c(5,3), linear.output = FALSE)

#predict
predicted.nn.values <- compute(nn, test[,1:30])

#round predicted values
predictions <- sapply(predicted.nn.values$net.result,round)

#table
g <-table(predictions, test$diagnosis)
g.t <- sum(diag(g))/sum(g)
print(g.t)

# VII. <a id='id11'>Accuracy</a>


accur<- matrix(c(c.t,d.t,e.t,f.t,g.t),ncol=1,byrow=FALSE)
colnames(accur) <- c("Accuracy")
rownames(accur) <- c("LG","RF","KNN","SVM","NN")
accur <- as.table(accur)
accur






Comments (0)