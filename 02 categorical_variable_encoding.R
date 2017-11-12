
# Load the library for using weight of evidence encoding
library(klaR)

# Importing the preprocessed train data i.e. train data after handling outliers and missing values
input=read.csv("data6_train.csv")

# Converting the response variable to a factor
input$cancel=as.factor(input$cancel)

#Converting all the categorical variables required to be encoded into factors
input$claim.ind=as.factor(input$claim.ind)
input$ni.marital.status=as.factor(input$ni.marital.status)
input$zip.code=as.factor(input$zip.code)

woemodel <- woe(cancel~., data = input, zeroadj=0.5, applyontrain = TRUE)

traindata_category_encoded<- predict(woemodel, input_categorical, replace = TRUE)

# Breaking train to encoded and unencoded variables
input_categorical=input[,c(1,2,3,4,5,6,7,12,13,15,19)]
input_numeric=input[,-c(1,2,3,4,5,6,12,13,15,19)]

#Computes weight of evidence transform of factor variables for binary classification 
woemodel <- woe(cancel~., data = input_categorical, zeroadj=0.5, applyontrain = TRUE)

traindata_category_encoded<- predict(woemodel, input_categorical, replace = TRUE)

#Merging back the data with encoded and unencoded columns 
library("dplyr")
traindata_encoded_all=left_join(traindata_category_encoded,input_numeric,by=c("id"="id"))


write.csv(traindata_encoded_all,'traindata_encoded_all_03Nov2017.csv')
