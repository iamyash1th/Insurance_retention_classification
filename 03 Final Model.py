
############################################### Refer to the file "categorical_variable_encoding.R" ########################################


############################# MODEL BUILDING WITH ENCODED FEATURES ####################


# The modified train and test files are named as yash_modified_train_10282017.csv and yash_modified_test_10282017.csv


### Model buliding after new encoding

train = pd.read_csv("yash_modified_train_10282017.csv")

y = train['cancel'] 

X = train.drop(['id','source','cancel'], axis = 1)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 


### Trying other gradient boosting models

from sklearn.ensemble import GradientBoostingClassifier
gbm= GradientBoostingClassifier()
kfold = KFold(n_splits= 10, random_state= 123)
result = cross_val_score(gbm, X, y, cv=kfold, scoring='roc_auc')
print(np.median(result))
print(result.mean())
#--0.73463844408
#-- 0.727299913133


from xgboost import XGBClassifier
xgb = XGBClassifier()
kfold = KFold(n_splits=20, random_state=123)
result = cross_val_score(xgb, X, y, cv=kfold, scoring='roc_auc')
print(np.median(result)
print(result.mean())
#--0.728562135201
#--0.726233697615



#***********************************K fold cross validation with different values of K such as 5,10,15,20,25
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#logreg.fit(X,y)
kfold = KFold(n_splits= 5, random_state= 123)
result = cross_val_score(logreg, X, y, cv=kfold, scoring='roc_auc')
print(result.mean())
print(np.median(result))

#0.73810398602
#0.731815156673


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()         
#logreg.fit(X,y)
kfold = KFold(n_splits= 10, random_state= 123)
result = cross_val_score(logreg, X, y, cv=kfold, scoring='roc_auc')
print(result.mean())
print(np.median(result))

#0.738898558566
#0.743736847224


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#logreg.fit(X,y)
kfold = KFold(n_splits= 15, random_state= 123)
result = cross_val_score(logreg, X, y, cv=kfold, scoring='roc_auc')
print(result.mean())
print(np.median(result))

#0.738253438705
#0.735635721908


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#logreg.fit(X,y)
kfold = KFold(n_splits= 20, random_state= 123)
result = cross_val_score(logreg, X, y, cv=kfold, scoring='roc_auc')
print(result.mean())  
print(np.median(result))

#0.737491404743
#0.731358407097            


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#logreg.fit(X,y)
kfold = KFold(n_splits= 25, random_state= 123)
result = cross_val_score(logreg, X, y, cv=kfold, scoring='roc_auc')
print(result.mean())
print(np.median(result))

#0.737553309317
#0.744791666667


############# Calculating Variance Inflation Factor (VIF) for each variable to diagnose multicollinearity########################
import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import scale

train = pd.read_csv("yash_modified_train_10282017.csv")
test = pd.read_csv("yash_modified_test_10282017.csv")

data=pd.concat([train, test],ignore_index=True)

y1 = data['cancel']

X1 = data.drop(['cancel','id','source'], axis =1)

vif = pd.DataFrame()
vif["features"] = X1.columns
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif.round(1)

data['ni.age'] = scale(data['ni.age'])
data['premium'] = scale(data['premium'])
data['len.at.res'] = scale(data['len.at.res'])



##Identifying columns with high VIF, combining them to a single variable as an average of scaled features
data['new_column'] = (data['len.at.res']+data['ni.age']+data['premium'])/3

train_modified = data.loc[data['source']=='train']
test_modified = data.loc[data['source']=='test']

train_modified.to_csv('train_modified_3_11.csv',index=False)
test_modified.to_csv('test_modified_3_11.csv',index=False)

train = pd.read_csv('train_modified_3_11.csv')
test = pd.read_csv('test_modified_3_11.csv')

y = train['cancel']
X_train = train.drop(['cancel','id','len.at.res','ni.age','premium','source'],axis =1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y)

X_test = test.drop(['id','len.at.res','ni.age','premium','source'], axis =1)

result = logreg.predict_proba(X_test)
pred = result[:,1]

test['pred'] = pred

test.to_csv("Submission_final_3_11_vif.csv",columns =['id','pred'],index = False)

