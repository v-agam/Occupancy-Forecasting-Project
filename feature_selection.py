# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 21:27:25 2021

@author: Legion
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
from statistics import mean

def select_features(month_table, list_of_communities, housing_contracts, units, housing_contracts_residents):
    print("Performing feature selection on all communities...")
    my_dict = {}
    mean_squared_error_list = []
    feature_imp_list = []
    
    for selected_community in list_of_communities:
        print(f"selected_community is {selected_community} ")
       
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
        if 'active_at' not in cols_selected:
            cols_selected.append('active_at')
        df = final_df[cols_selected]
        #print(cols_selected)
        occupancy_points = df.merge(month_table['last_day_of_month'], how = 'cross')
        
        occupancy_points['is_occupied'] = occupancy_points.apply(lambda row: (row['move_in_date'] <= row['last_day_of_month']
        and not pd.isna(row['move_in_date'])) and (pd.isna(row['move_out_date']) or row['move_out_date'] > row['last_day_of_month']), axis = 1)
        
        occupancy_points = occupancy_points[occupancy_points['is_occupied'] == True]
        occupancy_points = occupancy_points.drop(['is_occupied'], axis = 1)
        #occupancy_points['Number_of_prospects'] = 0
        #idx = 0
        #for date in occupancy_points.last_day_of_month:
        #occupancy_points['Number_of_prospects'] = occupancy_points[occupancy_points['active_at'] <= occupancy_points['last_day_of_month']]
        #    idx += 1
        
        occupancy_points = occupancy_points.groupby(['community_id','last_day_of_month'], as_index = False).sum()
        #print(occupancy_points.head())
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
        
        df_occupancy = occupancy_table.dropna(subset=['occupancy_points'])
        df1 = df_occupancy.copy()
        
        #print(df1['active_at'])
        ##print(df1.columns)
        df1 = df1.set_index('last_day_of_month')
        #print(df1.head())
        
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
            gbr.fit(train_features, train_labels);
            # Use the forest's predict method on the test data
            predictions = gbr.predict(test_features)
            # Calculate and display accuracy
            mse_gbr = mean_squared_error(test_labels, predictions)
            mean_squared_error_list.append(mse_gbr)
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
            
        
        else:
            print(f"Unable to apply ML algorithm for community {selected_community} due to very few data points")
    total_mse = mean(mean_squared_error_list)
    return my_dict, feature_imp_list, total_mse