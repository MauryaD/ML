# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:42:54 2019

@author: Deepak Maurya
"""
import numpy as np
import pandas as pd

#Query 1
dataset = pd.read_csv('DHFL.csv')       #reading the CSV file
dataset.Date.value_counts()         #counting the repetitions
dataset = dataset[dataset.Series == 'EQ']  
dataset.reset_index(inplace = True, drop = True)    #dropping the repetited Series.
#Query 1 completed


#Query 2: max, min and mean prices for last 90 days
data_max_price = dataset.iloc[-90:,8].max()
data_min_price = dataset.iloc[-90:,8].min()
data_mean_price = dataset.iloc[-90:,8].mean()
#Completed


#Query 3
dataset.dtypes          #returns data type as object
dataset['Date'] = pd.to_datetime(dataset['Date'], format = '%d-%b-%Y')      #date-type converted to datetime64[ns]
#Completed


#Query 4 VWAP
#VWAP_data = {'Months':[], 'Years':[]}           
#VWAP = pd.DataFrame(VWAP_data)                      #new DataFrame created 
dataset['Months'] = dataset.Date.str.slice(3,6)       #values for month added from previous dataset
dataset['Years'] = dataset.Date.str.slice(7)           ##values for Year added from previous dataset

temp_ds = dataset.groupby(['Months','Years'])

vwap = np.zeros(dataset.Months.unique().size)
for i, j in enumerate(dataset.Months.unique()):
    temp = dataset[dataset.Months == j]
    #temp_2 = dataset.groupby(['Months', 'Years']).reset_index(inplace = True)
    vwap[i] = ((temp["Total Traded Quantity"]*temp["Average Price"]).sum())/((temp["Total Traded Quantity"]).sum())
    
VWAP = pd.DataFrame(data = vwap,index = dataset.Months.unique())
VWAP.columns = ['VWAP']

#query5
def calAvg(ds, N):
    return ds.iloc[-N:,:]['Close Price'].mean()
def profitloss(df, N):
    print (np.round(100*(dataset.iloc[-1,:]['Close Price'] - dataset.iloc[-(N+1),:]['Close Price'])/dataset.iloc[-(N+1),:]['Close Price'],3),'%')
    return 
#Completed


#Query 6
dataset['Day_Perc_Change'] = 100*dataset['Close Price'].pct_change()
dataset.iloc[0,-1]=0

#Query 7
dataset['Trend'] = 0
for i in np.arange(dataset.Day_Perc_Change.size):
    if dataset.Day_Perc_Change[i]>=-0.5 and dataset.Day_Perc_Change[i]<0.5:
        dataset['Trend'][i] = 'Slight change or No Change'
        
    elif dataset.Day_Perc_Change[i]>=0.5 and dataset.Day_Perc_Change[i]<1:
        dataset['Trend'][i] ='Slight Positive'
        
    elif dataset.Day_Perc_Change[i]>=-1 and dataset.Day_Perc_Change[i]<-0.5:
        dataset['Trend'][i] ='Slight Negative'
        
    elif dataset.Day_Perc_Change[i]>=1 and dataset.Day_Perc_Change[i]<3:
        dataset['Trend'][i] ='Positive'
        
    elif dataset.Day_Perc_Change[i]>=-3 and dataset.Day_Perc_Change[i]<-1 :
        dataset['Trend'][i] ='Negative'
        
    elif dataset.Day_Perc_Change[i]>=3 and dataset.Day_Perc_Change[i]<7:
        dataset['Trend'][i] ='Among Top Gainers'
        
    elif dataset.Day_Perc_Change[i]>=-7 and dataset.Day_Perc_Change[i]<-3:
        dataset['Trend'][i] ='Among Top Loser'
        
    elif dataset.Day_Perc_Change[i]>7:
        dataset['Trend'][i] ='Bull Run'
        
    elif dataset.Day_Perc_Change[i]<-7:
        dataset['Trend'][i] ='Bear Drop'
#Completed

#query 8
New_Trend = dataset.groupby('Trend')['Total Traded Quantity'].agg(['mean','median'])
#completed
    
#QUery 9
dataset.to_csv('week2.csv', index = False)
#completed
    
    
    
    
    
    
    
    


    
    
    















