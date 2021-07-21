# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 22:49:15 2021

@author: Agam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from month_table_create import create_month_table
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE

def select_features_community(start_date1, end_date1, month_table, selected_community, housing_contracts, units, housing_contracts_residents):
    print("Performing feature selection on single community...")  
       
    #today = datetime.combine(date.today(), datetime.min.time())       
    housing_contracts1 = housing_contracts[housing_contracts['community_id'] == selected_community]
    units1 = units[units['community_id'] == selected_community]
    prospect_sub = housing_contracts_residents[housing_contracts_residents['community_id'] == selected_community]
    
    final_df = prospect_sub.merge(housing_contracts1, how = 'left', left_on = 'housing_contract_id', right_on = 'id')
    #print(final_df.columns)
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
    what = df.copy()
    occupancy_points = df.merge(month_table['last_day_of_month'], how = 'cross')
    
    occupancy_points['is_occupied'] = occupancy_points.apply(lambda row: (row['move_in_date'] <= row['last_day_of_month']
    and not pd.isna(row['move_in_date'])) and (pd.isna(row['move_out_date']) or row['move_out_date'] > row['last_day_of_month']), axis = 1)
    
    occupancy_points = occupancy_points[occupancy_points['is_occupied'] == True]
    occupancy_points = occupancy_points.drop(['is_occupied'], axis = 1)
    
    to_remove = []
    for ele in list(occupancy_points.columns):
        if 'id' in ele:
            if 'second_resident_market_rate' in ele:
                continue     
            else:
                to_remove.append(ele)
    
    cols = list(set(list((occupancy_points.columns.values))) - set(to_remove))
    if 'community_id' not in cols:
        cols.append('community_id')
    occupancy_points = occupancy_points[cols]
    
    occupancy_points = pd.get_dummies(occupancy_points, columns = ['current_caregiver', 'name','prospect_care_type_name'])
    
    
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
    
    start_date1 = datetime.strptime(start_date1,"%d/%m/%Y")
    end_date1 = datetime.strptime(end_date1,"%d/%m/%Y")
    occupancy_table_custom = occupancy_table[occupancy_table['last_day_of_month'] >= start_date1]
    occupancy_table_custom = occupancy_table_custom[occupancy_table_custom['last_day_of_month'] <= end_date1]
    
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
    
    #if 'month' in df1.columns:
        #df1 = pd.get_dummies(df1, columns = ['month'])
     #   df1 = df1.drop('month', axis = 1)
    #if 'year' in df1.columns:
        #df1["year"] = df1["year"].astype("category")

     #   df1 = df1.drop('year', axis = 1)
    
    
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
        feature_selector = RFE(gbr, n_features_to_select=2, step=1)
        feature_selector = feature_selector.fit(train_features, train_labels)
        predictions = feature_selector.predict(test_features)
        # Calculate and display accuracy
        mse_gbr = mean_squared_error(test_labels, predictions)
        support = feature_selector.support_
        indices = [i for i, x in enumerate(support) if x]
        selected_features = [feature_list[i] for i in indices]  
                
    
    else:
        print(f"Unable to apply ML algorithm for community {selected_community} due to very few data points")
   
    return df1, what, selected_features, occupancy_table, occupancy_table_custom, total_move_ins_custom_dates, total_move_outs_custom_dates