# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 20:29:22 2021

@author: Agam
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
from pmdarima.arima import auto_arima

def forecast_sarimax(var_to_predict, df, start_date, end_date):    
    
    model = auto_arima(var_to_predict, 
                          start_p=1,
                          start_q=1,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=True,    # No Seasonality
                          start_P=0, 
                          D=0, 
                          trace=True,
                          method = 'bfgs',
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True)    
    
    
    mod = sm.tsa.statespace.SARIMAX(var_to_predict,
                                    order = model.order,   #order=(1,1,1),  
                                    seasonal_order = model.seasonal_order,  #seasonal_order=(0,0,0,12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    
    results = mod.fit()
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()
    var_to_predict1 = var_to_predict.to_frame()
    v = var_to_predict1.columns[0]
    pred = results.get_prediction(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), dynamic=False)
    pred_ci = pred.conf_int()
    
    ax = var_to_predict[start_date:end_date].plot(label='Actual')
    pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.75)
    
    #ax.fill_between(pred_ci.index,
    #                pred_ci.iloc[:, 0],
    #                pred_ci.iloc[:, 1], color='k', alpha=.35)
    
    ax.set_xlabel('Date')
    plt.legend()
    plt.title(f"SARIMAX : {v}")
    plt.show()
    
    y_forecasted = pred.predicted_mean
    y_truth = var_to_predict[start_date:]

    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()

    return y_forecasted, mse