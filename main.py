# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:32:59 2021

@author: Agam
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime
from datetime import date
import warnings
from forecasting_prophet import forecast_prophet
from forecasting_arima import forecast_sarimax
from unite_data import merge_datasets
from month_table_create import create_month_table
from feature_selection import select_features
from community_features import select_features_community
from forecasting_varmax import forecast_varmax

#--------REQUIRED INPUTS BY USER---------------------------------------------------------------------------------------------------
start_date = "31/08/2018"  # Should be end date of any month, preferably after 2017 --- User to select
end_date = "31/08/2022"  # User to select
no_of_future_values = 12 # in order of months, starts from end_date--- User to select
selected_community = 1031  # User to select (except 200 and 309)
#------------------------------------------------------------------------------------------------------------------------------

#Do not display warnings 
warnings.filterwarnings("ignore")

#Read data from .csv files
communities = pd.read_csv(r"communities.csv")
prospects = pd.read_csv(r"prospects.csv")
housing_contracts = pd.read_csv(r"housing_contracts.csv")
units = pd.read_csv(r"units.csv")
floor_plans = pd.read_csv(r"floor_plans.csv")
housing_contracts_residents = pd.read_csv(r"housing_contracts_residents.csv")
residents = pd.read_csv(r"residents.csv")
prospect_stage = pd.read_csv(r"stages.csv")
prospect_score = pd.read_csv(r"scores.csv")
prospect_expected_move_timing = pd.read_csv(r"expected_move_timings.csv")
care_type =  pd.read_csv(r"care_types.csv")
close_reason =  pd.read_csv(r"close_reasons.csv")

# Merge data
communities, prospects, housing_contracts, units, floor_plans, housing_contracts_residents, residents, prospect_stage, prospect_score, prospect_expected_move_timing, care_type, close_reason = merge_datasets(communities, prospects, housing_contracts, units, floor_plans, housing_contracts_residents, residents, prospect_stage, prospect_score,
                   prospect_expected_move_timing, care_type, close_reason)

# Month and date selection for generating the complete time-series
month_table_ini = create_month_table(10, 2021) 

#Get list of community_id for all the existing communities
list_of_communities = list(housing_contracts.community_id.unique())
list_of_communities.remove(200) #Remove community with no entry for move_ins
list_of_communities.remove(309) #Remove community with no entry for move_ins

#Feature selection over all the communities
'''
my_dict, feature_imp_list, total_mse = select_features(month_table_ini, list_of_communities, housing_contracts, units, housing_contracts_residents)

plt.bar(range(len(my_dict)), list(my_dict.values()), align='center')
plt.xticks(range(len(my_dict)), list(my_dict.keys()), rotation=90, ha="right")
plt.tick_params(axis='x', which='major', labelsize=8)
plt.tight_layout()
plt.ylabel('Importance'); plt.xlabel('Variable'); 
plt.title('Variable Importances for determining move_ins - GBR');
plt.show()
'''

#Feature selection for a particular community
if selected_community in list_of_communities and len(housing_contracts[housing_contracts['community_id'] == selected_community]) > 4:
    
    df1, what, selected_features, occupancy_table, occupancy_table_custom, total_move_ins_custom_dates, total_move_outs_custom_dates = select_features_community(start_date, 
                            end_date, month_table_ini, selected_community, housing_contracts, units, housing_contracts_residents)


    if 'move_ins' not in selected_features:
        selected_features.append('move_ins')
    if 'move_outs' not in selected_features:
        selected_features.append('move_outs')
    if 'occupancy_points' not in selected_features:
        selected_features.append('occupancy_points')
    
    df = df1[selected_features]   
    df = df.sort_index()
    length = len(df)
    t_val = int(0.90*length)
    train = df.iloc[:t_val,:]
    test = df.iloc[t_val:,:]
    vars_to_predict = [df['move_ins'], df['move_outs'], df['occupancy_points']]
    var_to_predict = df['occupancy_points']
    
    
    '''
    Performing Time-series analysis
    '''
    
    from statsmodels.tsa.stattools import adfuller, kpss 
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    
    plt.plot(occupancy_table['last_day_of_month'], occupancy_table['move_ins'])
    plt.ylabel('Move-in occupancy points'); plt.xlabel('Date'); 
    plt.title(f"Move-in occupancy points over the years for Community {selected_community}");
    plt.xticks(rotation=50, ha="right")
    plt.show()
    plt.plot(occupancy_table['last_day_of_month'], occupancy_table['move_outs'])
    plt.ylabel('Move-out occupancy points'); plt.xlabel('Date');
    plt.title(f"Move-out occupancy points over the years for Community {selected_community}");
    plt.xticks(rotation=50, ha="right")
    plt.show()
    plt.plot(occupancy_table['last_day_of_month'], occupancy_table['occupancy_points'])
    plt.ylabel('Total occupancy points'); plt.xlabel('Date');
    plt.title(f"Total occupancy points over the years for Community {selected_community}");
    plt.xticks(rotation=50, ha="right")
    plt.show()   
       
    # Testing for stationarity of Time-series
    
    var_to_predict = df['occupancy_points']
    
    result = adfuller(var_to_predict.values, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
    
    result = kpss(var_to_predict.values, regression='c')
    print('\nKPSS Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[3].items():
        print('Critial Values:')
        print(f'   {key}, {value}')     
    
    # Checking for helping features for the time-series data forecasting
    
    from statsmodels.tsa.stattools import grangercausalitytests
    
    for col in df.columns:
        print(f"predictor is {col}")
        grangercausalitytests(df[['occupancy_points', col]], maxlag=2)
    
    size = int(len(var_to_predict)/2)-1
    fig, axes = plt.subplots(1,2,figsize=(16,7), dpi= 100)
    plot_acf(var_to_predict.tolist(), lags=size, ax=axes[0])
    plot_pacf(var_to_predict.tolist(), lags=size, ax=axes[1])
    plt.show()
    
    #DIFFERENCING
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(var_to_predict); axes[0, 0].set_title('Original Series')
    plot_acf(df.move_ins, ax=axes[0, 1])
    
    # 1st Differencing
    axes[1, 0].plot(var_to_predict.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(var_to_predict.diff().dropna(), ax=axes[1, 1])
    
    # 2nd Differencing
    axes[2, 0].plot(var_to_predict.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(var_to_predict.diff().diff().dropna(), ax=axes[2, 1])
    plt.show()
    
    
    '''
    Forecasting and Back-testing using different methods
    '''
    
    vars_to_predict = [df['move_ins'], df['move_outs'], df['occupancy_points']]
    
    # Prophet
    pred_prophet = []
    mse_prophet = []
    for var in vars_to_predict:    
        f_prophet = forecast_prophet(var, no_of_future_values)
        pred_prophet.append(f_prophet)
        inter = var.to_frame()
        inter = inter.reset_index()
        mse = ((f_prophet['yhat'][:-no_of_future_values] - inter.iloc[:,1]) ** 2).mean()
        mse_prophet.append(mse)
    
    
    #SARIMAX ---ARIMA with seasonality
    pred_sarimax=[]
    mse_sarimax=[]
    df1 = df.diff().dropna()
    
    for var in vars_to_predict:    
        y_forecasted, mse = forecast_sarimax(var, df, start_date, end_date)
        mse_sarimax.append(mse)
        pred_sarimax.append(y_forecasted)
    
    
    
    #VARMAX
    pred_varmax, mse_varmax = forecast_varmax(df, vars_to_predict, start_date, end_date)


else:
    print(f"The selected community_id {selected_community} does not have enough contracts/data points to perform a reasonable analysis")





