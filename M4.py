# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:32:49 2019

@author: Deepak Maurya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

#1.a) 
data = pd.read_csv('week3.csv')
data.dropna(inplace = True)
data.reset_index(inplace = True)

data['Call'] = 0
for i in np.arange(data.Average.size):
    if data['Average Price'][i] <= data.LowerBand[i]:
        data['Call'][i] = 'Buy'   
    elif data['Average Price'][i] >= data.UpperBand[i]:
        data['Call'][i] = 'Short'
    elif data['Average Price'][i] > data.LowerBand[i] and data['Average Price'][i] < data.Average[i]:
        data['Call'][i] = 'Hold Buy/ Liquidate Short'
    else:
        data['Call'][i] = 'Hold Short/ Liquidate Buy'
        
#1.b)
from sklearn.model_selection import train_test_split
RFX = data[['LowerBand', 'Average', 'UpperBand', 'Average Price']]
RFY = data['Call']

RFX_train, RFX_test, RFY_train, RFY_test = train_test_split(RFX, RFY, test_size = 1/3, random_state = 24)

from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators = 200, criterion= 'entropy', max_depth = 10, n_jobs = -1, random_state = 30,)
classifier.fit(RFX_train, RFY_train)

RFY_pred = classifier.predict(RFX)
plt.scatter(RFY, RFY_pred)
plt.show()

check = [RFY.values, RFY_pred]
check = pd.DataFrame(check)
check = check.T
check.columns = ['Calls', 'Predictions']

#for mis-match 
f = 0
for i in np.arange(len(data.Average)):
    if check.iloc[i,0] != check.iloc[i,1]:
        f = f+1
print(f)

#for accuracy
accuracy = ((RFY_test.size - f)/ RFY_test.size) *100

#1.c)

df = pd.read_csv('APOLLOTYRE.csv')

def BollingerBand(price, len = 14, numsd = 2):
    avg = price.rolling(len).mean()
    sd =  price.rolling(len).std()
    upband = avg + (sd*numsd)
    lowband = avg - (sd*numsd)
    return np.round(lowband,3), np.round(avg,3), np.round(upband,3)

df['LowerBand'], df['Average'], df['UpperBand'] = BollingerBand(df['Close Price'])

df['Average Price'].plot(color = 'k', lw = 2., figsize = (20,10))
df['Average'].plot(color = 'b', figsize = (20,10), lw = 1.)
df['UpperBand'].plot(color = 'g', lw = 1., figsize = (20,10))
df['LowerBand'].plot(color = 'r', lw = 1., figsize = (20,10))
plt.show()

#training the new model
df.dropna(inplace = True)
df.reset_index(inplace = True)

df['Call'] = 0
for i in np.arange(df.Average.size):
    if df['Average Price'][i] <= df.LowerBand[i]:
        df['Call'][i] = 'Buy'   
    elif df['Average Price'][i] >= df.UpperBand[i]:
        df['Call'][i] = 'Short'
    elif df['Average Price'][i] > df.LowerBand[i] and df['Average Price'][i] < df.Average[i]:
        df['Call'][i] = 'Hold Buy/ Liquidate Short'
    else:
        df['Call'][i] = 'Hold Short/ Liquidate Buy'

#X = df[['LowerBand', 'Average', 'UpperBand', 'Average Price']]
y = df['Call']
RFX_train, RFX_test, y_train, y_test = train_test_split(RFX, y, test_size = 1/3, random_state = 24)
classifier.fit(RFX_train, y_train)
y_pred = classifier.predict(RFX)

plt.scatter(y, y_pred)
plt.show()

check1 = [y.values, y_pred]
check1 = pd.DataFrame(check1)
check1 = check1.T
check1.columns = ['Calls', 'Predictions']

#for mis-match 
f1 = 0
for i in np.arange(len(data.Average)):
    if check1.iloc[i,0] != check1.iloc[i,1]:
        f1 = f1+1
print(f1)

accuracy1 = ((y_test.size - f1)/ y_test.size) *100

#1 Completed

dataset = pd.read_csv('ITC.csv')
#since Series = EQ and BL, therefore I have to remove them
dataset = dataset[dataset.Series == 'EQ']
dataset.reset_index(inplace = True, drop = True)

dataset['Perc_chng_Open-Close'] = ((dataset['Open Price'] - dataset['Close Price'])/dataset['Open Price'])*100
dataset['Perc_chng_HL'] = ((dataset['High Price'] - dataset['Low Price'])/dataset['High Price']) * 100

dataset['Day_Perc_chng'] = 100* dataset['Close Price'].pct_change()
dataset.iloc[0,-1] = 0
dataset['roll_mean'] = dataset['Day_Perc_chng'].rolling(5).mean()
dataset['roll_std'] = dataset['Day_Perc_chng'].rolling(5).std()
dataset.dropna(inplace = True)

dataset['Action'] = np.where(dataset['Close Price'].shift(-1)>dataset['Close Price'], 1, -1)

#Classification Model
RFx = dataset[['Perc_chng_Open-Close','Perc_chng_HL', 'roll_mean', 'roll_std']]
RFy = dataset['Action']

RFx_train, RFx_test, RFy_train, RFy_test = train_test_split(RFx, RFy, test_size = 1/3, random_state = 0)

clf = RandomForestClassifier(random_state = 5, criterion = 'entropy', n_estimators = 100)
model = clf.fit(RFx_train, RFy_train)

from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(RFy_test, model.predict(RFx_test), normalize = True) * 100.0 )

dataset['Strategy_returns'] = dataset.Day_Perc_chng * model.predict(RFx)

dataset.Strategy_returns[RFy_train.size:].hist()
plt.xlabel('Strategy Returns %')
plt.show()

#net cumulative returns (in %)
((dataset.Strategy_returns[RFy_train.size:]+100)/100).cumprod().plot()
plt.ylabel('Strategy Returns %')
plt.show()
