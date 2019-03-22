#!/usr/bin/python3.6  
# -*- coding: utf-8 -*-
import pandas as pd
from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

#load the data

#file='D:\\data\\international-airline-passengers.csv'
file='D:\data\\amzna.csv'
data = pd.read_csv(file)
#train=data[:int(0.7*(len(data)))]
#valid=data[int(0.7*(len(data))):]
#dateparse=parse('01-02-2010',dayfirst=True)
#dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%y')

#data['Date'] = pd.to_datetime(data['Date'],format='%d-%m-%y')
#data=data.sort_values(by='Date')
#divide into train and validation set

train=data[:-5]
valid=data[-5:]

 
#preprocessing (since arima takes univariate series as input)
#train.drop('Month',axis=1,inplace=True)
#valid.drop('Month',axis=1,inplace=True)
 
#plotting the data
train['Close'].plot()
valid['Close'].plot()
#train['International airline passengers'].plot()
#valid['International airline passengers'].plot()

model=auto_arima(train['Close'], trace=True,error_action='ignore', suppress_warnings=True,seasonal=False)
#model=ARIMA(train['Close'],order=(2,1,2))
model.fit(train['Close'])

forecast=model.predict(n_periods=len(valid))
forecast=pd.DataFrame(forecast,index=valid.index,columns=['Prediction'])
 
#plot the predictions for validation set
plt.plot(train['Close'], label='Train')
plt.plot(valid['Close'], label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()

pre=[801.3575404761137,784.5456688118664,786.3877932627745,799.1078987929279,787.1475481092913]
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid['Close'],forecast))
rms2= sqrt(mean_squared_error(pre,forecast))
print(rms)

