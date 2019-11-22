# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:51:11 2019

@author: Deepak Maurya
"""

import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


df = pd.read_csv('week2.csv')

df['Date'] = pd.to_datetime(df['Date'])  

#Q1
df.set_index('Date')
df.index = df.Date
df['Close Price'].plot(figsize= (10,5))

#Q2
fig = plt.figure(num = 'B', figsize = (10,5))
plt.stem(df.Date, df['Day_Perc_Change'])

#Q3
fig = plt.figure(num = 'C', figsize = (10,5))
plt.stem(df.Date, df['Day_Perc_Change'])
(df['Total Traded Quantity']/5000000).plot(figsize = (10,5))

#Q4
pie_data = df.groupby('Trend').Trend.count()
pie_data.plot.pie(figsize = (20,15), autopct = '%1.1f%%')

bar_data = df.groupby('Trend')['Total Traded Quantity'].agg(['mean', 'median'])
#bar_data.plot.bar(figsize = (20,15))
bar_data.plot.bar(figsize = (20,15), subplots = True)       #subplots -- creates separate graphs

#Q5
df.Day_Perc_Change.hist(figsize = (30,15),bins = 75)
#plt.show()

#Q6
df1 = pd.read_csv('IDFC.csv')
df2 = pd.read_csv('ITDC.csv')           #Loading 5 new stocks
df3 = pd.read_csv('NCC.csv')
df4 = pd.read_csv('PVR.csv')
df5 = pd.read_csv('FORTIS.csv')

df1 = df1[df1.Series == 'EQ']
df1.reset_index(inplace = True, drop = True)

'''df2 has no other Equitity than EQ
hence no changing in series of df2
checked by -- df2.Series.unique() function and df2.Series.value_counts() func'''

df3 = df3[df3.Series == 'EQ']
df3.reset_index(inplace = True , drop = True)

df4 = df4[df4.Series == 'EQ']
df4.reset_index(inplace = True , drop = True)           #Dropping Equitities'''
            
df5 = df5[df5.Series == 'EQ']
df5.reset_index(inplace = True , drop = True)

df1 = df1[['Close Price']]
df1.columns = ['IDFC']

df2 = df2[['Close Price']]
df2.columns = ['ITDC']

df3 = df3[['Close Price']]              #keeping close prices and renaming the column names
df3.columns = ['NCC']

df4 = df4[['Close Price']]
df4.columns = ['PVR']

df5 = df5[['Close Price']]
df5.columns = ['FORTIS']

new_df = pd.concat([df1, df2, df3, df4, df5], axis = 1)    #new dataframe for all the closing prices with col name as company's name
Perc_Chng_new_df = 100*new_df.pct_change()              #calculating the %change   
Perc_Chng_new_df = Perc_Chng_new_df.replace([np.inf, -np.inf], np.nan)
Perc_Chng_new_df = Perc_Chng_new_df.dropna()            #deleting all the NaN values

import seaborn as sns;
sns.pairplot(Perc_Chng_new_df)

'''#from scipy.stats import linregress
#linregress(df1,df5)
from scipy import stats
a = np.array([Perc_Chng_new_df['ITDC']])
b = np.arange(494)
stats.pearsonr(a.any(),b)

for i in Perc_Chng_new_df.columns:
    sns.jointplot(i, 'IDFC',Perc_Chng_new_df, kind = 'scatter')
''' 
#Q6 Completed

#Q7
volatility = Perc_Chng_new_df['NCC'].rolling(7).std()*np.sqrt(7)
volatility.plot(figsize= (20,10))
plt.show()
# COmpleted

#Q8 
volatility1 = Perc_Chng_new_df[['NCC', 'ITDC', 'PVR']].rolling(7).std()*np.sqrt(7)
volatility1.plot(figsize= (20,10))
plt.show()
#completed

#Q9
signal = pd.DataFrame(index = df.index)
signal['signal'] = 0.0              #sell = 0 nd buy = 1
#SMA for 21 days
signal['21_SMA'] = df['Close Price'].rolling(window = 21, min_periods = 1).mean()
#SMA for 34 days
signal['34_SMA'] = df['Close Price'].rolling(window = 34, min_periods = 1).mean()

signal['signal'][21:] = np.where(signal['21_SMA'][21:]>signal['34_SMA'][21:], 1,0)
signal["Position"] = signal['signal'].diff()            #1 for buy and -1 to sell

#ploting the SMAs
fig = plt.figure(figsize = (20,10))
ax1 = fig.add_subplot(111, ylabel = 'Price of stock')
df['Close Price'].plot(ax = ax1, lw = 2., color = 'r')
signal[['21_SMA','34_SMA']].plot(ax= ax1, lw = 1.)
ax1.plot(signal.loc[signal.Position == 1.0].index, signal['21_SMA'][signal.Position == 1.0],'^', markersize = 20, color = 'g')
ax1.plot(signal.loc[signal.Position == -1.0].index, signal['21_SMA'][signal.Position == -1.0],'v', markersize = 20, color = 'k')
plt.show()
#completed

#Q10 Trase calls using Bollinger Bands
def Bband(price, len = 14, numsd = 2):
    avg = price.rolling(len).mean()
    sd =  price.rolling(len).std()
    upband = avg + (sd*numsd)
    downband = avg - (sd*numsd)
    return np.round(avg,3), np.round(upband,3), np.round(downband,3)
df['Average'], df['UpperBand'], df['DownBand'] = Bband(df['Close Price'])
df['Average Price'].plot(color = 'k', lw = 2., figsize = (20,10))
df['Average'].plot(color = 'b', figsize = (20,10), lw = 1.)
df['UpperBand'].plot(color = 'g', lw = 1., figsize = (20,10))
df['DownBand'].plot(color = 'r', lw = 1., figsize = (20,10))
plt.show()

#df.drop('UpperBande', axis = 1, inplace = True)

