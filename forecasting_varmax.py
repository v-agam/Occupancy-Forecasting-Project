# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:02:00 2021

@author: Legion
"""
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt

def forecast_varmax(df, vars_to_predict, start_date, end_date):
        
    model = VARMAX(df)#, order=(0, 0)) 
    model_fit = model.fit() #maxlags = 1, method = 'newton'
    pred_varmax = model_fit.get_prediction(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), dynamic=False)
    forecast_varmax = pred_varmax.predicted_mean
    
    pred_ci = pred_varmax.conf_int()
    pred_varmax=[]
    mse_varmax = []
    
    for var in vars_to_predict:
        ax = var[start_date:end_date].plot(label='Actual')
        var = var.to_frame()
        v = var.columns[0]
        forecast_varmax[v].plot(ax=ax, label='Forecast', alpha=.75)
        ax.set_xlabel('Date')
        plt.legend()
        plt.title(f"VARMAX : {v}")
        plt.show()
        y_fore= forecast_varmax[v]
        pred_varmax.append(y_fore)
        y_fore = y_fore.to_frame()
        y_truth = var[start_date:]
        # Compute the mean square error
        mse = ((y_fore - y_truth) ** 2).mean()
        mse_varmax.append(mse.sum())
    
    return pred_varmax, mse_varmax