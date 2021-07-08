# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:32:59 2021

@author: Agam
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import numpy as np
from statistics import mean
from datetime import datetime
from datetime import date
import calendar
import seaborn as sns
from itertools import chain
from scipy.stats import norm
from xgboost import XGBRegressor
 
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
'''
Merging Units and floor_plans into Units
'''
units = units.merge(floor_plans, how = 'left', left_on = 'floor_plan_id', right_on = 'id')
units = units.drop(columns = ['id_y','community_id_y'])
units = units.rename(columns = {'occupancy_points': 'floor_plan_occupancy_points', 'id_x': 'id', 'created_at_x':'Unit_created_at',
                                'updated_at_x':'Unit_updated_at', 'created_at_y':'floor_plan_created_at', 'updated_at_y':'floor_plan_updated_at'
                                ,'community_id_x':'community_id', 'market_rate':'floor_plan_market_rate'})
units = units[units['off_census'] == 'f'] #remove off census units

'''
Merging housing_contracts and Units into housing_contracts
'''
housing_contracts = housing_contracts.merge(units, how = 'left', left_on = 'unit_id', right_on = 'id')
housing_contracts = housing_contracts.drop(columns = ['id_y','community_id_y'])
housing_contracts = housing_contracts.rename(columns = {'id_x': 'id','community_id_x':'community_id',
                                                        'created_at':'Housing_contract_created_at','updated_at':'housing_contract_updated_at'})
#convert move-in and move-out dates to datetime
housing_contracts['move_in_date'] = pd.to_datetime(housing_contracts['move_in_date'], errors = 'coerce')
housing_contracts['move_out_date'] = pd.to_datetime(housing_contracts['move_out_date'], errors = 'coerce')
# Calculate Occupancy points per contract
housing_contracts['occupancy_points'] = housing_contracts['occupancy_point_factor'] * housing_contracts['floor_plan_occupancy_points']

'''
Merging Prospects, prospect_stage, prospect_expected_move_timing, prospect_score and Communities
'''
prospects = prospects.merge(prospect_stage, how = 'left', left_on = 'stage_id', right_on = 'id')
prospects = prospects.rename(columns = {'id_x': 'prospect_id', 'position':'prospect_stage_position'})
prospects = prospects.merge(close_reason, how = 'left', left_on = 'close_reason_id', right_on = 'id')
prospects = prospects.merge(communities, how = 'left', left_on = 'community_id', right_on = 'id')
prospects = prospects.drop(columns = ['id_y','updated_at_y','created_at_y'])
prospects = prospects.rename(columns = {'name':'close_reason_name', 'created_at_x':'prospect_created_at','updated_at_x':'prospect_updated_at', 'status':'prospect_status',
                                        'account_id_x':'prospect_account_id','account_id_y':'community_account_id', })
prospects = prospects.merge(prospect_expected_move_timing, how = 'left', left_on = 'expected_move_timing_id', right_on = 'id')
prospects = prospects.rename(columns = {'score':'expected_move_timing_score', 'position':'expected_move_timing_position'})
prospects = prospects.merge(prospect_score, how = 'left', left_on = 'score_id', right_on = 'id')
prospects = prospects.rename(columns = {'position':'prospect_score_position'})

'''
Merging residents and housing_contracts_residents
'''
residents = residents.merge(care_type, how = 'left', left_on = 'care_type_id', right_on = 'id')
residents = residents.rename(columns = {'name': 'prospect_care_type_name', 'id_x': 'id'})
housing_contracts_residents = housing_contracts_residents.merge(residents, how = 'left', left_on = 'resident_id', right_on = 'id')
'''
Merging prospects and housing_contracts_residents
'''
housing_contracts_residents = housing_contracts_residents.merge(prospects, how = 'left', left_on = 'prospect_id', right_on = 'prospect_id')


def create_month_table(month, year):
    """
    Create shell table
    """
    months = []
    years = []
    last_days_of_month = []

    #fill in months and years
    for i in range(144):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        months.append(month)
        years.append(year)

    #calculate last day of each month (for occupancy calculation)
    for i, j in zip(months, years):
        last_day_of_month = calendar.monthrange(j, i)[1]
        last_day_of_month_date = str(j)+'-'+str(i)+'-'+str(last_day_of_month)
        last_day_of_month_date = datetime.strptime(last_day_of_month_date, '%Y-%m-%d')
        last_days_of_month.append(last_day_of_month_date)

    month_table = pd.DataFrame({'month': months, 'year': years, 'last_day_of_month': last_days_of_month})
    return month_table


month_table = create_month_table(10, 2021) # Month and date selection for generating the complete time-series
feature_imp_list = []
list_of_communities = list(housing_contracts.community_id.unique())
list_of_communities.remove(200) #Remove community with no entry for move_ins
list_of_communities.remove(309)
'''
for selected_community in list_of_communities:
    housing_contracts1 = housing_contracts[housing_contracts['community_id'] == selected_community] 
    if housing_contracts1.move_in_date.isnull().all():
         notnull_community.append(selected_community)
'''

my_dict = {}
mean_squared_error_list = []

#for selected_community in list_of_communities:
#print(f"selected_community is {selected_community} ")
   
#today = datetime.combine(date.today(), datetime.min.time())
selected_community = 1
   
housing_contracts1 = housing_contracts[housing_contracts['community_id'] == selected_community]
units1 = units[units['community_id'] == selected_community]
prospect_sub = housing_contracts_residents[housing_contracts_residents['community_id'] == selected_community]

final_df = prospect_sub.merge(housing_contracts1, how = 'left', left_on = 'housing_contract_id', right_on = 'id')
final_df = final_df.drop(columns = ['occupancy_point_factor','floor_plan_occupancy_points','care_type_id_y', 'community_id_y', 'id_y','sales_counselor_id_y'])
final_df = final_df.rename(columns = {'id_x': 'id','care_type_id_x':'care_type_id', 'community_id_x':'community_id','sales_counselor_id_x':'sales_counselor_id'})

def removemissingvaluecol(dff, threshold):
    col_list = []
    col_list = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index)) >= threshold))].columns, 1).columns.values)
    return col_list

#df = final_df[['community_id', 'move_in_date', 'move_out_date', 'occupancy_points','square_feet','move_out_reason_id', 'floor_plan_market_rate', 
                       #'prospect_stage_position','system_type', 'prospect_score_position', 'expected_move_timing_score']]

cols_selected = removemissingvaluecol(final_df,95)  # Remove columns with greater than 95% missing values
if 'move_out_date' not in cols_selected:
    cols_selected.append('move_out_date')
if 'move_in_date' not in cols_selected:
    cols_selected.append('move_in_date')
if 'occupancy_points' not in cols_selected:
    cols_selected.append('occupancy_points')
df = final_df[cols_selected]

occupancy_points = df.merge(month_table['last_day_of_month'], how = 'cross')

occupancy_points['is_occupied'] = occupancy_points.apply(lambda row: (row['move_in_date'] <= row['last_day_of_month']
and not pd.isna(row['move_in_date'])) and (pd.isna(row['move_out_date']) or row['move_out_date'] > row['last_day_of_month']), axis = 1)

occupancy_points = occupancy_points[occupancy_points['is_occupied'] == True]
occupancy_points = occupancy_points.drop(['is_occupied'], axis = 1)
occupancy_points = occupancy_points.groupby(['community_id','last_day_of_month'], as_index = False).sum()
occupancy_table = month_table.merge(occupancy_points, how = 'left', on = ['last_day_of_month'])
community_total_occupancy_points = units1[['community_id', 'floor_plan_occupancy_points']].groupby(['community_id'], as_index = False).sum()
community_total_occupancy_points = community_total_occupancy_points.rename(columns = {'floor_plan_occupancy_points': 'total_occupancy_points'})
occupancy_table = occupancy_table.merge(community_total_occupancy_points, how = 'left', on = ['community_id'])
occupancy_table['occupancy_percentage'] = occupancy_table['occupancy_points'] / occupancy_table['total_occupancy_points']

# PART B - Calculation of total move-ins and move-outs in a given time period

month_table_start = create_month_table(9, 2021)  # month should be one less than that for month_table_end mentioned below
month_table_end = create_month_table(10, 2021)  # exactly same as month_table created for occupancy point calculations
total_move_ins_in_month = []
total_move_outs_in_month = []

for start_date, end_date in zip(month_table_start.last_day_of_month, month_table_end.last_day_of_month):
    move_calc = housing_contracts1[['community_id', 'move_in_date', 'move_out_date', 'count_move_in','lease_canceled_on', 
                                      'occupancy_points','count_move_out']]
    
    move_calc['move_in'] = move_calc.apply(lambda row: row['move_in_date'] <= end_date and row['move_in_date'] >= start_date and
                     pd.isna(row['lease_canceled_on']) and row['count_move_in'] == 't', axis = 1)
    
    move_calc['move_out'] = move_calc.apply(lambda row: row['move_out_date'] <= end_date and row['move_out_date'] >= start_date
                                           and row['count_move_out'] == 't', axis = 1)
    
    
    move_calc = move_calc.merge(month_table['last_day_of_month'], how = 'cross')
    move_calc1 = move_calc[move_calc['move_in'] == True]
    move_calc2 = move_calc[move_calc['move_out'] == True]
    
    move_calc1 = move_calc1.drop(['move_in'], axis = 1)
    move_calc2 = move_calc2.drop(['move_out'], axis = 1)
    
    move_calc1 = move_calc1.groupby(['community_id','last_day_of_month'], as_index = False).sum()
    move_calc2 = move_calc2.groupby(['community_id','last_day_of_month'], as_index = False).sum()
    
    if move_calc1.empty:
        total_move_ins_in_month.append(0)
    else:
        total_move_ins_in_month.append(move_calc1.occupancy_points.iloc[0])
    
    if move_calc2.empty:
        total_move_outs_in_month.append(0)
    else:
        total_move_outs_in_month.append(move_calc2.occupancy_points.iloc[0])

# Appending move-ins and move-outs to occupancy table
occupancy_table['move_ins'] = total_move_ins_in_month
occupancy_table['move_outs'] = total_move_outs_in_month

# Finding relevant data for user-input dates
start_date = "13/04/2019"
end_date = "13/5/2020"
start_date = datetime.strptime(start_date,"%d/%m/%Y")
end_date = datetime.strptime(end_date,"%d/%m/%Y")
occupancy_table_custom = occupancy_table[occupancy_table['last_day_of_month'] >= start_date]
occupancy_table_custom = occupancy_table_custom[occupancy_table_custom['last_day_of_month'] <= end_date]

total_move_ins_custom_dates = occupancy_table_custom.move_ins.sum()
total_move_outs_custom_dates = occupancy_table_custom.move_outs.sum()
df_occupancy = occupancy_table.dropna(subset=['occupancy_points'])
df1 = df_occupancy.copy()
#df1['Number_of_prospects'] = 0
df1 = df1.set_index('last_day_of_month')
#for idx in df1.index:
 #   df1[idx, 'Number_of_prospects'] = df1[df1[idx]]


# Remove id's from the data

to_remove = []
for ele in list(df1.columns):
    if 'id' in ele:
        if 'second_resident_market_rate' in ele:
            continue     
        else:
            to_remove.append(ele)

cols = list(set(list((df1.columns.values))) - set(to_remove))
df1 = df1[cols]

if 'month' in df1.columns:
    df1 = df1.drop('month', axis = 1)
if 'year' in df1.columns:
    df1 = df1.drop('year', axis = 1)
#if 'move_outs' in df1.columns:
    #df1 = df1.drop('move_outs', axis = 1)
    
if len(df1) >= 4:
    labels = np.array(df1['move_ins'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features = df1.drop(['move_ins', 'move_outs'], axis = 1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
    features = np.array(features)
    
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    
    #XGBoost = XGBRegressor().fit(train_features, np.array(train_labels))
    gbr = GradientBoostingRegressor(random_state=0) 
    #rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=0, n_jobs= -1)
    # Train the model on training data
    feature_selector = RFE(gbr, n_features_to_select=7, step=1)
    feature_selector = feature_selector.fit(train_features, train_labels)
    #gbr.fit(train_features, train_labels);
    # Use the forest's predict method on the test data
    predictions = feature_selector.predict(test_features)
    # Calculate and display accuracy
    mse_rf = mean_squared_error(test_labels, predictions)
    mean_squared_error_list.append(mse_rf)
    support = feature_selector.support_
    indices = [i for i, x in enumerate(support) if x]
    selected_features = [feature_list[i] for i in indices]   
    

else:
    print(f"Unable to apply ML algorithm for community {selected_community} due to very few data points")



'''
# Get numerical feature importances - LOOP only

total_mse = mean(mean_squared_error_list)
plt.bar(range(len(my_dict)), list(my_dict.values()), align='center')
plt.xticks(range(len(my_dict)), list(my_dict.keys()), rotation=90, ha="right")
plt.tick_params(axis='x', which='major', labelsize=8)
plt.tight_layout()
plt.ylabel('Importance'); plt.xlabel('Variable'); 
plt.title('Variable Importances for determining move_outs - RF');
plt.show()

importances = list(gbr.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    feature_imp_list.append(feature_importances)
    # List of tuples with variable and importance
    for f in feature_importances:
        if f[0] in my_dict:
            my_dict[f[0]] += f[1]
        if f[0] not in my_dict:
            my_dict[f[0]] = f[1]

'''
    
'''
Performing Time-series analysis
'''

plt.plot(occupancy_table['last_day_of_month'], occupancy_table['move_ins'])
plt.ylabel('Move-in occupancy points'); plt.xlabel('Date'); 
plt.title(f"Move-in occupancy points over the years for Community {selected_community}");
plt.show()
plt.plot(occupancy_table['last_day_of_month'], occupancy_table['move_outs'])
plt.ylabel('Move-out occupancy points'); plt.xlabel('Date');
plt.title(f"Move-out occupancy points over the years for Community {selected_community}");
plt.show()  
    
# Testing for stationarity of Time-series
 
if 'move_ins' not in selected_features:
    selected_features.append('move_ins')
if 'move_outs' not in selected_features:
    selected_features.append('move_outs')
df = df1[selected_features]

from statsmodels.tsa.stattools import adfuller, kpss    

result = adfuller(df.move_outs.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

result = kpss(df.move_outs.values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')    
    

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df.move_ins.tolist(), lags=50, ax=axes[0])
plot_pacf(df.move_ins.tolist(), lags=50, ax=axes[1])


from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})
df_ma = df.move_ins.rolling(3, center=True, closed='both').mean()
df_ma.plot()

#DIFFERENCING
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.move_ins); axes[0, 0].set_title('Original Series')
plot_acf(df.move_ins, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.move_ins.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.move_ins.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.move_ins.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.move_ins.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df.move_ins, order=(1,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

model_fit.plot_predict(dynamic=False)
plt.show()

from statsmodels.tsa.stattools import acf

# Create Training and Test
#df = df.diff()
train = df.move_ins[30:]
test = df.move_ins[:30]

model = ARIMA(train, order=(1, 1, 1))  
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(30, alpha=0.5)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

 
import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(df.move_ins,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()





import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

df = df.sort_index()
data_d = df.diff().dropna()

train = data_d.iloc[:-10,:]
test = data_d.iloc[-10:,:]

forecasting_model = VAR(train)
results = model.fit(maxlags=15, ic='aic')
lag_order = results.k_ar
results.forecast(df.values[-lag_order:], 5)





