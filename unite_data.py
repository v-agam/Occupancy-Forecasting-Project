# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 21:12:43 2021

@author: Agam
"""
import pandas as pd

def merge_datasets(communities, prospects, housing_contracts, units, floor_plans, housing_contracts_residents, residents, prospect_stage, prospect_score,
                   prospect_expected_move_timing, care_type, close_reason):
    
    print("Merging different datasets...")
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
    prospects['active_at'] = pd.to_datetime(prospects['active_at'], errors = 'coerce')
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
    
    return communities, prospects, housing_contracts, units, floor_plans, housing_contracts_residents, residents, prospect_stage, prospect_score, prospect_expected_move_timing, care_type, close_reason