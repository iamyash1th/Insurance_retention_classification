import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
%matplotlib inline

# Reading the train file
train = pd.read_csv("train.csv")

## Removing -1 from cancel column
train = train[data.cancel != -1]

train.shape

#(7553, 18)

train.to_csv("train1.csv", index = False)


## train1 is the file with -1 removed from cancel column

train = pd.read_csv("train1.csv")

train.cancel.value_counts()

#0    5710
#1    1843

# Reading the test file
test = pd.read_csv("Test.csv")

#####################################################################################################################3

## We are performing covariate shift analysis to see if train and test datasets belong to the same distribution and whether they 
# can be combined together for analysis

import numpy as np
import pandas as pd
from pandas import Series, trainFrame
import matplotlib.pyplot as plt
% matplotlib inline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

## Outlier and Missing value for train.
## Missing values are replaced with mode for categorical variables and with median for numeric variables
## Outliers are replaced with median

for i in train.columns:
    if i in ["claim.ind","ni.gender","ni.marital.status","sales.channel","coverage.type","dwelling.type","credit","house.color","zip.code"]:        
       train[i] = train[i].fillna(train[i].mode()[0])
    if i in ["tenure","n.adults","n.children","premium","len.at.res","ni.age"]:
       train[i] = train[i].fillna(train[i].median())
	   
	   
train.isnull().sum()
--0

for i in test.columns:
    if i in ["claim.ind","ni.gender","ni.marital.status","sales.channel","coverage.type","dwelling.type","credit","house.color","zip.code"]:        
       test[i] = test[i].fillna(test[i].mode()[0])
    if i in ["tenure","n.adults","n.children","premium","len.at.res","ni.age"]:
       test[i] = test[i].fillna(test[i].median())
	   
test.isnull().sum()
--0

Q1 = train['n.adults'].quantile(0.25)
Q3 = train['n.adults'].quantile(0.75)
IQR = Q3 - Q1
median_value = train['n.adults'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

train['n.adults'] = train['n.adults'].apply(imputer)

Q1 = test['n.adults'].quantile(0.25)
Q3 = test['n.adults'].quantile(0.75)
IQR = Q3 - Q1
median_value = test['n.adults'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

test['n.adults'] = test['n.adults'].apply(imputer)


Q1 = train['n.children'].quantile(0.25)
Q3 = train['n.children'].quantile(0.75)
IQR = Q3 - Q1
median_value = train['n.children'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

train['n.children'] = train['n.children'].apply(imputer)

## Outlier and Missing value for test


Q1 = train['len.at.res'].quantile(0.25)
Q3 = train['len.at.res'].quantile(0.75)
IQR = Q3 - Q1
median_value = train['len.at.res'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

train['len.at.res'] = train['len.at.res'].apply(imputer)



Q1 = train['ni.age'].quantile(0.25)
Q3 = train['ni.age'].quantile(0.75)
IQR = Q3 - Q1
median_value = train['ni.age'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

train['ni.age'] = train['ni.age'].apply(imputer)


Q1 = train['premium'].quantile(0.25)
Q3 = train['premium'].quantile(0.75)
IQR = Q3 - Q1
median_value = train['premium'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

train['premium'] = train['premium'].apply(imputer)



number = LabelEncoder()
for i in train.columns:
    if (train[i].dtype == 'object'):
      train[i] = number.fit_transform(train[i].astype('str'))
      train[i] = train[i].astype('object')

for i in test.columns:
    if (test[i].dtype == 'object'):
      test[i] = number.fit_transform(test[i].astype('str'))
      test[i] = test[i].astype('object')
	  
	  
	  
## creating a new feature origin
train['origin'] = 0
test['origin'] = 1

   
training = train.drop('cancel',axis=1) #droping target variable


## taking sample from training and test train
training = training.sample(2000, random_state=19)
testing = test.sample(2000, random_state=17)

## combining random samples
combi = training.append(testing)
y = combi['origin']
combi.drop('origin',axis=1,inplace=True)


## modelling
model = RandomForestClassifier(n_estimators = 50, max_depth = 5,min_samples_leaf = 5)

drop_list = []
for i in combi.columns:
    score = cross_val_score(model,pd.dataFrame(combi[i]),y,cv=10,scoring='roc_auc')
    if (np.mean(score) > 0.8):
        drop_list.append(i)
		
print(i,np.mean(score))
# -- No drifitng features
# All features are important and train and test can be analyzed together as they come from the same distribution



##############################################################################################################

## Appending the original train and test datasets and replacing NA's and  treating outliers using aforementioned logic

train['source']= 'train'
test['source'] = 'test'
data=pd.concat([data, test],ignore_index=True)
data.shape
#9965, 19

## data.csv - File with -1 removed from cancel column and train and test combined together 

# Tenure - median
#claim.ind - mode  replace them with 0
# n.adults - replace with mode
# n.children - replace with mode


data['zip.code'] =  data['zip.code'].fillna(data['zip.code'].mode()[0])
data.to_csv("data1.csv", index = False)


# EDA 

# Replacing missing values and removing outliers

## TENURE COLUMN

# Nulls

data.tenure.isnull().sum()
#--4

data.tenure.mean()

data.tenure.median()

## Imputing with median

data['tenure'] = data['tenure'].fillna(data['tenure'].median())

data.tenure.isnull().sum()
#--0

# Outliers

data.tenure.plot("box")

Q1 = data['tenure'].quantile(0.25)
Q3 = data['tenure'].quantile(0.75)
IQR = Q3 - Q1
median_value = data['tenure'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

data['tenure'] = data['tenure'].apply(imputer)


data.tenure.plot("box")


## claim.ind


data['claim.ind'].isnull().sum()
#--12

# Imputing with mode

data['claim.ind'] = data['claim.ind'].fillna(data['claim.ind'].mode()[0])

data['claim.ind'].isnull().sum()
#--0



## n.adults

data['n.adults'].isnull().sum()
#--9

data['n.adults'] = data['n.adults'].fillna(data['n.adults'].median())

data['n.adults'].isnull().sum()
#--0

data['n.adults'].plot("box")

## Removing outliers

Q1 = data['n.adults'].quantile(0.25)
Q3 = data['n.adults'].quantile(0.75)
IQR = Q3 - Q1
median_value = data['n.adults'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

data['n.adults'] = data['n.adults'].apply(imputer)


data['n.adults'].plot("box")
-- Outliers cleared



## n.children


data['n.children'].isnull().sum()
#--1

data['n.children'] = data['n.children'].fillna(data['n.children'].median())

data['n.children'].isnull().sum()
#--0

data['n.children'].plot("box")

## Removing outliers

Q1 = data['n.children'].quantile(0.25)
Q3 = data['n.children'].quantile(0.75)
IQR = Q3 - Q1
median_value = data['n.children'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

data['n.children'] = data['n.children'].apply(imputer)


data['n.children'].plot("box")
#-- 2 remaining




## Gender


data['ni.gender'].isnull().sum()
#--13

# Imputing with mode

data['ni.gender'] = data['ni.gender'].fillna(data['ni.gender'].mode()[0])

data['ni.gender'].isnull().sum()
#--0

# Marital status
data['ni.marital.status'] = data['ni.marital.status'].fillna(data['ni.marital.status'].mode()[0])
# Sales channel
data['sales.channel'] = data['sales.channel'].fillna(data['sales.channel'].mode()[0])
#coverage type
data['coverage.type'] = data['coverage.type'].fillna(data['coverage.type'].mode()[0])
#dwelling type
data['dwelling.type'] = data['dwelling.type'].fillna(data['dwelling.type'].mode()[0])


## len at res

data['len.at.res'].isnull().sum()
#--8

data['len.at.res'] = data['len.at.res'].fillna(data['len.at.res'].median())

data['len.at.res'].isnull().sum()
#--0

data['len.at.res'].plot("box")

## Removing outliers

Q1 = data['len.at.res'].quantile(0.25)
Q3 = data['len.at.res'].quantile(0.75)
IQR = Q3 - Q1
median_value = data['len.at.res'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

data['len.at.res'] = data['len.at.res'].apply(imputer)

data['len.at.res'].plot("box")
#-- cleaned

# credit
data['credit'] = data['credit'].fillna(data['credit'].mode()[0])

#house.color
data['house.color'] = data['house.color'].fillna(data['house.color'].mode()[0])


# ni age

data['ni.age'] = data['ni.age'].fillna(data['ni.age'].median())


Q1 = data['ni.age'].quantile(0.25)
Q3 = data['ni.age'].quantile(0.75)
IQR = Q3 - Q1
median_value = data['ni.age'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

data['ni.age'] = data['ni.age'].apply(imputer)

data['ni.age'].plot("box")


## Zip code

data['zip.code'] = data['zip.code'].fillna(data['zip.code'].mode()[0])

data['zip.code'].isnull().sum()


# Premium

data['premium'] = data['premium'].fillna(data['premium'].median())


Q1 = data['premium'].quantile(0.25)
Q3 = data['premium'].quantile(0.75)
IQR = Q3 - Q1
median_value = data['premium'].median()

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

def imputer(value):
    if value < lower_limit or value > upper_limit:
        return median_value
    else:
        return value

data['premium'] = data['premium'].apply(imputer)

data['premium'].plot("box")


## Rounding off to 2 dgits
data['len.at.res'] = round(data['len.at.res'],2)
data['premium'] = round(data['premium'],2)

                                                            
data.to_csv("data6.csv", index = False)


## data6.csv is the file with all NA's and Outliers are replaced with median 


## Numerical encoding


from sklearn.preprocessing import LabelEncoder   
le = LabelEncoder()

var_to_encode = ["ni.gender"]

data['ni.gender'] = le.fit_transform(data['ni.gender'])


## One hot encoding

## Creating dummy varaibles

var_to_encode = ["sales.channel","coverage.type","dwelling.type","credit","house.color","state"]

data = pd.get_dummies(data, columns = var_to_encode, drop_first = True)

data.to_csv("data7.csv")  


#********************************************************************

## Seperating train and test files

train_modified = data.loc[data['source']=='train']
test_modified = data.loc[data['source']=='test']

train_modified.to_csv('train_modified.csv',index=False)
test_modified.to_csv('test_modified.csv',index=False)


#*********************************MODEL BUILDING******************************

## Base model

train = pd.read_csv("train_modified.csv")

y = train['cancel']

X = train.drop(["cancel","id","year","source","zip.code"], axis=1)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 


# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,y)
kfold = KFold(n_splits= 20, random_state= 147)
result = cross_val_score(logreg, X, y, cv=kfold, scoring='roc_auc')
print(result)
#Train cross validated median -- 71.3%


#[ 0.72403632  0.72878181  0.75403065  0.72655978  0.67455243  0.69103651
# 0.74935401  0.72325581  0.74391722  0.72495567  0.66160609  0.7533826
#  0.72436466  0.71467205  0.68396726  0.69125705  0.71102227  0.75720275
#  0.70038715  0.66958106]
  
  
# 2. Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbm= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
kfold = KFold(n_splits= 20, random_state= 147)
result = cross_val_score(gbm, X, y, cv=kfold, scoring='roc_auc')
print(result)


#[0.71937719  0.71501394  0.72198447  0.71017776  0.67075163  0.68086701
 # 0.73129845  0.72009044  0.74594754  0.72299793  0.66336067  0.74869251
  #0.72587914  0.68971655  0.66504574  0.67921031  0.70907101  0.73075551
  #0.71312427  0.65491803]
  
# 3.XGB 

from xgboost import XGBClassifier
xgb = XGBClassifier()
kfold = KFold(n_splits=20, random_state=147)
result = cross_val_score(xgb, X, y, cv=kfold, scoring='roc_auc')
print(result)

#[ 0.72144791  0.7224207   0.74990048  0.72165824  0.67362887  0.68917862
 # 0.73620801  0.71417959  0.72336837  0.74368351  0.66520759  0.76269528
  #0.71372636  0.69611803  0.65426095  0.67445608  0.70490672  0.74110249
  #0.70514905  0.66388889]
  
  
# Logistic Regression gave good scores

## Improving our base model
  
 
## Categorical variables encoded with weight of evidence method and house color dropped 


############################################### Refer to the file "02 categorical_variable_encoding.R" ########################################




