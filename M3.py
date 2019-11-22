# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:43:34 2019

@author: Deepak Maurya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('GOLD.csv')
'''
import statsmodels.api as sm
x = data[['Price', 'Open', 'High', 'Low']]
y = data['new']
x1 = sm.add_constant(x)
model = sm.OLS(y,x1)
result = model.fit()
print(result.summary())'''

dummy_df = data.dropna()

X = dummy_df[['Price', 'Open', 'High', 'Low']]
y = dummy_df['Pred']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)          #predicted Values

plt.scatter(y_test, y_pred, c = ['r'])

data['Pred'] = regressor.predict(data[['Price', 'Open', 'High', 'Low']])            #adding the predicted values
sns.distplot(y_test-y_pred, bins = 5)           #frequency of the error values
    
#---------------------------------------------------------------
import statsmodels.api as sm
df = pd.read_csv('CIPLA.csv')
df1 = pd.read_csv('Nifty50.csv')

df = df[df.Series == 'EQ']
df.index = df["Date"]
df1.index = df1.Date

df = df[['Close Price']]
df.columns = ['CIPLA']

df1 = df1[['Close']]
df1.columns = ['Nifty50']

prices = pd.concat([df,df1], axis =1)

returns = prices.pct_change()
returns = returns.dropna(axis=0)

returns = returns.iloc[-60:,:]

X = returns['Nifty50']
y = returns['CIPLA']
#adding a const
x1= sm.add_constant(X)

#model
model = sm.OLS(y,x1)

#fitting the result
result = model.fit()
print(result.summary())

##Monthly BETA
prices['months'] = prices.index.str.slice(3)
month = np.zeros((25,2))
for i,j in enumerate (prices.months.unique()):
    temp = prices[prices.months == j]
    month[i] = temp.iloc[-1,0:2]

month = pd.DataFrame(month)
month.columns = ['Cipla', 'Nifty']
month.index = prices.months.unique()        #month's last closing price 

monthly_returns = month.pct_change()
monthly_returns = monthly_returns.dropna(axis=0)

Mx = monthly_returns['Nifty']
My = monthly_returns['Cipla']

Mx1 = sm.add_constant(Mx)
model = sm.OLS(My,Mx1)
Mresult = model.fit()
print(result.summary())
#complted!
