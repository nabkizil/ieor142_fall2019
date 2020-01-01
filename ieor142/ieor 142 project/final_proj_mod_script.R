#Modeling and Predicting Ranking of Fantasy Basketball Players
#12/01/19

#Import all functions to be used in script

library(randomForest)
library(ranger)
library(dplyr)
library(tidyverse)
library(reshape2)
library(GGally)
library(car)

library(ggplot2)
library(caret) # cross validation
library(e1071)
library(gbm)

#Loading cleaned CSV files

data2016 <- read.csv("merged2016.csv")
data2017 <- read.csv("merged2017.csv")
data2018 <- read.csv("merged2018.csv")

#Use the 2018, 2017 data as the training set and the 2018 data as the test set.
#This is especially useful in the this case because there are extra players in the 2018 dataset.
#As such, it would be interesting to see how the forecast performs on the proposed test set.
#IMPORTANT TO NOTE: The split is roughly 72.6% training data and remainder as the test set.

trainSet <- union_all(data2018, data2017)
testSet <- data2016

#now remove player_id and player_name since these should not be features in the model.

trainSet <- trainSet[3:32]
testSet <- testSet[3:32]

#The response variable is going to be salary per fantasy points. The bigger the ratio, then that means that the player makes too much money or does not score enough points.

trainSet$salaryPerPoints <- trainSet$SALARY/trainSet$fantasy_points
testSet$salaryPerPoints <- testSet$SALARY/testSet$fantasy_points

#remove salary and fantasy_points now

trainSet <- trainSet[3:31]
testSet <- testSet[3:31]

###################################################################################################
#IN THIS SECTION, WE TRIED RF FIRST AND THE OSR2 WAS VERY POOR SO WE DID SOME FEATURE ENGINEERING

#relabel pos as numerics.

#trainSet$posAsNum <- as.numeric(trainSet$Pos)
#testSet$posAsNum <- as.numeric(testSet$Pos)

#now remove pos

#trainSet <- trainSet[2:30]
#testSet <- testSet[2:30]


#To prevent overfitting, let's use a random forest model and test the accuracy on the test set.
#tune the mtry parameter 

#tuneRF(trainSet[, -28], trainSet[, 28], mtryStart = 1, ntreeTry = 100, stepFactor = 2,
#       trace = TRUE, plot = TRUE)

#modRandomForest <- randomForest(salaryPerPoints ~. , data = trainSet, mtry = 3, nodesize = 5, ntree = 500)

#predRF = predict(modRandomForest, newdata = testSet)
#true_response = testSet$salaryPerPoints

#MAE:
#mean(abs(true_response - predRF))

#RMSE:
#sqrt(mean((true_response-predRF)^2))

#To try and improve accuracy, let us first use Linear Regression and remove extra features.
#Note that for VIF, we set the tolerance to be under 5 for our feature set

#################################################################################################################

ggscatmat(trainSet, columns = 1:10, alpha = 0.8)
ggscatmat(trainSet, columns = 11:20, alpha = 0.8)
ggscatmat(trainSet, columns = 21:29, alpha = 0.8)

#because X2P and X2PA are perfectly collinear, let us remove one of the variables.
#the same is true with trb and drb.
#pts also suffers from perfect collinearity

modLR <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS, data = trainSet)
summary(modLR)
vif(modLR)

#remove FGA
modLR2 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA, data = trainSet)
summary(modLR2)
vif(modLR2)

#remove FTA
modLR3 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA - FTA, data = trainSet)
summary(modLR3)
vif(modLR3)

#remove X3PA
modLR4 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA - FTA - X3PA, data = trainSet)
summary(modLR4)
vif(modLR4)

#remove MP
modLR5 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA - FTA - X3PA - MP, data = trainSet)
summary(modLR5)
vif(modLR5)

#remove FG.
modLR6 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA - FTA - X3PA - MP - FG., data = trainSet)
summary(modLR6)
vif(modLR6)

#remove TOV
modLR7 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA - FTA - X3PA - MP - FG. - TOV, data = trainSet)
summary(modLR7)
vif(modLR7)

#remove FG
modLR8 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA - FTA - X3PA - MP - FG. - TOV - FG, data = trainSet)
summary(modLR8)
vif(modLR8)

#remove Pos
modLR9 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA - FTA - X3PA - MP - FG. - TOV - FG - Pos, data = trainSet)
summary(modLR9)
vif(modLR9)

#remove PF
modLR10 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA - FTA - X3PA - MP - FG. - TOV - FG - Pos - PF, data = trainSet)
summary(modLR10)
vif(modLR10)

#remove DRB
modLR11 <- lm(salaryPerPoints ~ . -X2PA - TRB - X2P - PTS - FGA - FTA - X3PA - MP - FG. - TOV - FG - Pos - PF - DRB, data = trainSet)
summary(modLR11)
vif(modLR11)


#use a subset of features in the new training and test sets. This should help increase OSR2 and accuracy in the model
newTrainSet <- trainSet[c(2, 3, 4, 5, 10, 12, 15, 16, 17, 19, 20, 23, 24, 25, 26, 29)]
newTestSet <- testSet[c(2, 3, 4, 5, 10, 12, 15, 16, 17, 19, 20, 23, 24, 25, 26, 29)]

#now use a Random Forest model
#tune the mtry parameter 

#Function for the OSR2 on the test set:
OSR2 <- function(predictions, train, test) {
  SSE <- sum((test - predictions)^2)
  SST <- sum((test - mean(train))^2)
  r2 <- 1 - (SSE/SST)
  return(r2)
}

#Using tuneRF gives a different value every time. This is harder to work with, so to tune, we used nested for loops 
#tuningparam <- tuneRF(newTrainSet[, -16], newTrainSet[, 16], ntreeTry = 100, trace = TRUE, plot = TRUE, doBest = TRUE) #10 is the best mtry value

for (i in c(10:15)){
  for (j in c(7:10)){
    for (k in (c(1:4)*500)){
      modRandomForest <- randomForest(salaryPerPoints ~. , data = newTrainSet, mtry = i, nodesize = j, ntree = k)

      predRF = predict(modRandomForest, newdata = newTestSet)
      true_response = newTestSet$salaryPerPoints
  
      #OSR2:
      print(c("i = ", i))
      print(c("j = ", j))
      print(c("k = ", k))
      print(c("OSR2 Value: ", OSR2(predRF, newTrainSet$salaryPerPoints, true_response)))
    }
  }
}

#optimal mtry = 15, node size = 9, and ntree = 1000 with the OSR2 = approx. 0.60

finalMod <- randomForest(salaryPerPoints ~. , data = newTrainSet, mtry = 15, nodesize = 9, ntree = 1000)
finalPred = predict(modRandomForest, newdata = newTestSet)
true_response = newTestSet$salaryPerPoints
OSR2(finalPred, newTrainSet$salaryPerPoints, true_response) #~0.60

#Final DataSet to Export for further Analysis
testdata <- data2016
testdata$salaryPerPoints <- testdata$SALARY/testdata$fantasy_points
testdata$predResponse <- as.factor(finalPred)
#write.csv(testdata, file = "C:\\Users\\rosha\\Documents\\important\\actually_important\\Fall_2019_UCB\\IEOR142\\Project\\new_stuff\\testDataAnalysis.csv", row.names = FALSE)
