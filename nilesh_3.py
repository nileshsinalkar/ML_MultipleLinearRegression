# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 13:50:07 2019

@author: niles
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# converts a data into dataframe
dataset=pd.read_csv('50_Startups.csv')
# converts a data in to array
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])

#converts numpy array into dataframe
x_df=pd.DataFrame(data=X,columns=['a','b','c','d'])
# create dummies for particular columns  of dataframe
x_df_dum=pd.get_dummies(data=x_df,columns =['d'])
#again create dataframe into numpy array
X_main=x_df_dum.values
#removind the dummy variable trap
X_use=X_main[:,:-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_use,Y,test_size=0.25,random_state=2)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


#backwardelimination


#import statsmodels.formula.api as sm

import statsmodels.api as sm

X_use=np.append(arr=np.ones((50,1)).astype(int),values=X_use,axis=1)

#building the efficient model

X_opt=X_use[:,[0,1,2,3,4,5]]

regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()

regressor_OLS_2=regressor_OLS_1.fit()
regressor_OLS_2.summmary()

