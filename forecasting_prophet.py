# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 20:34:12 2021

@author: Legion
"""

from prophet import Prophet

def forecast_prophet(var, no_of_future_values):
    my_model = Prophet()
    var = var.to_frame()
    v = var.columns[0]
    var = var.reset_index()
    var.columns = ['ds', 'y']
    my_model.fit(var)
    future_dates = my_model.make_future_dataframe(periods=no_of_future_values, freq='M', include_history=True)
    forecast = my_model.predict(future_dates)
    output_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]  
    my_model.plot(forecast, xlabel='Date', ylabel=f"Prophet : {v}")   
    my_model.plot_components(forecast)
    return output_forecast