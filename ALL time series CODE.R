#REVIEW 1
library(forecast)
CARSALES=c(1000,900,850,1050,1100,1200,1150,1120,1000)
CARSALES.ts=ts(CARSALES,frequency=4, start=c(2017,1))
autoplot(CARSALES.ts, xlab = "Year", ylab = "Sales (LAKHS)")
# USING THE SEASONAL_naive() METHOD to forecast CAR sales in 2020
CARSALES.fc=snaive(CARSALES.ts, h=4)
# PLOTTING and SUMMARIZING the forecasts
autoplot(CARSALES.fc)
summary(CARSALES.fc)


CARSALES=c(1000,900,850,1050,1100,1200,1150,1120,1000)
CARSALES.ts=ts(CARSALES,frequency=4, start=c(2017,2))
#Use naive() METHOD to forecast SALES in 2020
CARSALES.fc=naive(CARSALES.ts, h=4)
# PLOTTING and SUMMARIZING the forecasts
autoplot(CARSALES.fc)
summary(CARSALES.fc)

#MAIN
library(caret)
names(getModelInfo())

# Load data from Hadley Wickham on Github - Vehicle data set and predict 6 cylinder vehicles
library(RCurl)
#urlData <- getURL('https://raw.githubusercontent.com/hadley/fueleconomy/master/data-raw/vehicles.csv')
#vehicles <- read.csv(text = urlData)

# alternative way of getting the data
urlfile <-'https://raw.githubusercontent.com/hadley/fueleconomy/master/data-raw/vehicles.csv'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
vehicles <- read.csv(textConnection(x))

# clean up the data and only use the first 30 columns
vehicles <- vehicles[names(vehicles)[1:30]]
vehicles <- data.frame(lapply(vehicles, as.character), stringsAsFactors=FALSE)
vehicles <- data.frame(lapply(vehicles, as.numeric))
vehicles[is.na(vehicles)] <- 0
vehicles$cylinders <- ifelse(vehicles$cylinders == 6, 1,0)

prop.table(table(vehicles$cylinders))

# shuffle and split the data into three parts
set.seed(1234)
vehicles <- vehicles[sample(nrow(vehicles)),]
split <- floor(nrow(vehicles)/3)
ensembleData <- vehicles[0:split,]
blenderData <- vehicles[(split+1):(split*2),]
testingData <- vehicles[(split*2+1):nrow(vehicles),]

# set label name and predictors
labelName <- 'cylinders'
predictors <- names(ensembleData)[names(ensembleData) != labelName]

library(caret)
# create a caret control object to control the number of cross-validations performed
myControl <- trainControl(method='cv', number=3, returnResamp='none')

# quick benchmark model 
test_model <- train(blenderData[,predictors], blenderData[,labelName], method='gbm', trControl=myControl)
preds <- predict(object=test_model, testingData[,predictors])

library(pROC)
auc <- roc(testingData[,labelName], preds)
print(auc$auc) # Area under the curve: 0.9896

# train all the ensemble models with ensembleData
model_gbm <- train(ensembleData[,predictors], ensembleData[,labelName], method='gbm', trControl=myControl)
model_rpart <- train(ensembleData[,predictors], ensembleData[,labelName], method='rpart', trControl=myControl)
model_treebag <- train(ensembleData[,predictors], ensembleData[,labelName], method='treebag', trControl=myControl)

# get predictions for each ensemble model for two last data sets
# and add them back to themselves
blenderData$gbm_PROB <- predict(object=model_gbm, blenderData[,predictors])
blenderData$rf_PROB <- predict(object=model_rpart, blenderData[,predictors])
blenderData$treebag_PROB <- predict(object=model_treebag, blenderData[,predictors])
testingData$gbm_PROB <- predict(object=model_gbm, testingData[,predictors])
testingData$rf_PROB <- predict(object=model_rpart, testingData[,predictors])
testingData$treebag_PROB <- predict(object=model_treebag, testingData[,predictors])

# see how each individual model performed on its own
auc <- roc(testingData[,labelName], testingData$gbm_PROB )
print(auc$auc) # Area under the curve: 0.9893

auc <- roc(testingData[,labelName], testingData$rf_PROB )
print(auc$auc) # Area under the curve: 0.958

auc <- roc(testingData[,labelName], testingData$treebag_PROB )
print(auc$auc) # Area under the curve: 0.9734

# run a final model to blend all the probabilities together
predictors <- names(blenderData)[names(blenderData) != labelName]
final_blender_model <- train(blenderData[,predictors], blenderData[,labelName], method='gbm', trControl=myControl)

# See final prediction and AUC of blended ensemble
preds <- predict(object=final_blender_model, testingData[,predictors])
auc <- roc(testingData[,labelName], preds)
print(auc$auc)  # Area under the curve: 0.9922

