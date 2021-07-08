import pandas as pd
from datetime import datetime
from datetime import date
import calendar
import string

#Import some files, (I'm ignoring some datatype errors, like some IDs being read in as floats)
communities = pd.read_csv(r"communities.csv")
prospects = pd.read_csv(r"prospects.csv")
housing_contracts = pd.read_csv(r"housing_contracts.csv")
units = pd.read_csv(r"units.csv")
floor_plans = pd.read_csv(r"floor_plans.csv")

#Map communities onto prospects
def map_communities_onto_prospects(prospects, communities):
    prospects = prospects.merge(communities, how = 'left', left_on = 'community_id', right_on = 'id')
    print(prospects)
    return

def clean_units_and_contracts(floor_plans, units, housing_contracts):
    #clean units
    units = units.merge(floor_plans[['id', 'occupancy_points']], how = 'left', left_on = 'floor_plan_id', right_on = 'id') #map floor type, number of beds, and occupancy points
    units = units.drop(columns = ['id_y'])
    units = units.rename(columns = {'occupancy_points': 'floor_plan_occupancy_points', 'id_x': 'id'})
    units = units[units['off_census'] == 'f'] #remove off census units

    #################
    #clean housing_contracts
    #################
    #map occupancy points
    housing_contracts = housing_contracts.merge(units[['id', 'floor_plan_occupancy_points']], how = 'left', left_on = 'unit_id', right_on = 'id')

    #convert move-in and move-out dates to datetime
    housing_contracts['move_in_date'] = pd.to_datetime(housing_contracts['move_in_date'], errors = 'coerce')
    housing_contracts['move_out_date'] = pd.to_datetime(housing_contracts['move_out_date'], errors = 'coerce')

    #calculate contract occupancy points
    housing_contracts['occupancy_points'] = housing_contracts['occupancy_point_factor'] * housing_contracts['floor_plan_occupancy_points']
    
    #calculate month and year for move_in and move_out
    housing_contracts['move_in_month'] = pd.DatetimeIndex(housing_contracts['move_in_date']).month.fillna(0).astype('int32')
    housing_contracts['move_in_year'] = pd.DatetimeIndex(housing_contracts['move_in_date']).year.fillna(0).astype('int32')
    housing_contracts['move_out_month'] = pd.DatetimeIndex(housing_contracts['move_out_date']).month.fillna(0).astype('int32')
    housing_contracts['move_out_year'] = pd.DatetimeIndex(housing_contracts['move_out_date']).year.fillna(0).astype('int32')

    #reassigning count_move_in to account for cancelled leases
    housing_contracts['count_move_in'] = housing_contracts.apply(lambda row: pd.isna(row['lease_canceled_on']) and row['count_move_in'] == 't', axis = 1)
    housing_contracts['count_move_out'] = housing_contracts['count_move_out'] == 't'
    return units, housing_contracts

def calculate_eom_occupancy(housing_contracts, units, month_table):
    #filter to one community for the sake of visualization
    selected_community = 144
    housing_contracts = housing_contracts[housing_contracts['community_id'] == selected_community]
    units = units[units['community_id'] == selected_community]

    #count occupancy points
    eom_occupancy_points = housing_contracts[['community_id', 'move_in_date', 'move_out_date', 'occupancy_points']]
    eom_occupancy_points = eom_occupancy_points.merge(month_table['last_day_of_month'], how = 'cross')
    eom_occupancy_points['is_occupied'] = eom_occupancy_points.apply(lambda row: (row['move_in_date'] <= row['last_day_of_month'] 
    and not pd.isna(row['move_in_date'])) and (pd.isna(row['move_out_date']) or row['move_out_date'] > row['last_day_of_month']), axis = 1)
    eom_occupancy_points = eom_occupancy_points[eom_occupancy_points['is_occupied'] == True]
    eom_occupancy_points = eom_occupancy_points.drop(['is_occupied'], axis = 1)
    eom_occupancy_points = eom_occupancy_points.groupby(['community_id', 'last_day_of_month'], as_index = False).sum()
    occupancy_table = month_table.merge(eom_occupancy_points, how = 'left', on = ['last_day_of_month'])

    #count total occupancy points
    community_total_occupancy_points = units[['community_id', 'floor_plan_occupancy_points']].groupby(['community_id'], as_index = False).sum()
    community_total_occupancy_points = community_total_occupancy_points.rename(columns = {'floor_plan_occupancy_points': 'total_occupancy_points'})
    occupancy_table = occupancy_table.merge(community_total_occupancy_points, how = 'left', on = ['community_id'])

    #calculate occupancy %
    occupancy_table['occupancy_percentage'] = occupancy_table['occupancy_points'] / occupancy_table['total_occupancy_points']
    print(occupancy_table)
    return occupancy_table

def create_month_table():
    """
    Create shell table
    """
    months = []
    years = []
    last_days_of_month = []
    #today = date.today()
    month = 4    #today.month - 1
    year = 2021 #today.year

    #fill in months and years
    for i in range(24):
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

month_table = create_month_table()
units, housing_contracts = clean_units_and_contracts(floor_plans, units, housing_contracts)
eom_occupancy = calculate_eom_occupancy(housing_contracts, units, month_table)
#map_communities_onto_prospects(prospects, communities)


#CRAP BEGINS

'''
Missing data
'''
total1 = housing_contracts.isnull().sum().sort_values(ascending=True)
percent1 = (housing_contracts.isnull().sum()/housing_contracts.isnull().count()).sort_values(ascending=True)
missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total', 'Percent'])
missing_data1

# Data imputation
housing_contracts['occupancy_points'].fillna(0,inplace=True)

total2 = prospects.isnull().sum().sort_values(ascending=True)
percent2 = (prospects.isnull().sum()/prospects.isnull().count()).sort_values(ascending=True)
missing_data2 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])
missing_data2
#prospects = prospects.drop((missing_data[missing_data['Percent'] > 0.5]).index,1)
    
'''
Outlier data values identification and removal
'''
df = housing_contracts.drop(['move_in_date', 'move_out_date', 'created_at', 'updated_at','created_at_x', 'updated_at_x', 'created_at_y',
                             'updated_at_y','leased_on',  'lease_canceled_on', 'name', 'off_census','number','move_out_reason_details',
                             'count_move_in', 'count_move_out', 'occupancy_point_factor', 'floor_plan_occupancy_points'], axis=1)

col_names = ['id', 'community_id', 'unit_id', 'move_out_reason_id', 'sales_counselor_id', 'transfer_from_id', 'community_id_x',
             'floor_plan_id','care_type_id','community_id_y',  'default_privacy_level_id']

for col in col_names:
    df[col] = df[col].astype('category',copy=False)

df1 = df.dropna(subset=['beds']) #Impute?

df2 = df1.drop(['default_privacy_level_id','sales_counselor_id','unit_market_rate','square_feet','move_out_reason_id','monthly_rate',
                'community_fee_amount','deposit_amount', 'transfer_from_id', 'one_time_concession', 'recurring_concession', 'floor'],axis=1)
#df['move_out_reason_id'].fillna(0,inplace=True)
# Dropping the columns with very large empty rows > 90%

'''
Converting id columns into categorical variables
'''

#df.fillna(0,inplace=True)
outlier_list = IsolationForest(random_state=10).fit_predict(df2)

'''
Checking Correlation
'''
cm = df2.corr()
sns.set(font_scale=1.0)
hm = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=True, xticklabels=True)
plt.show()


'''
total3 = df2.isnull().sum().sort_values(ascending=True)
percent3 = (df2.isnull().sum()/df2.isnull().count()).sort_values(ascending=True)
missing_data3 = pd.concat([total3, percent3], axis=1, keys=['Total', 'Percent'])
missing_data3
'''
'''
Prospects table purification
'''
total3 = prospects.isnull().sum().sort_values(ascending=True)
percent3 = (prospects.isnull().sum()/prospects.isnull().count()).sort_values(ascending=True)
missing_data3 = pd.concat([total3, percent3], axis=1, keys=['Total', 'Percent'])
missing_data3
df_temp = prospects.dropna(subset=['occupancy_goal_percent', 'likelihood_to_move_from_home','familiarity_with_community','score_id',
                                   'stage_id', 'sales_counselor_id'])

df_temp = df_temp.drop(['active_at','locked_at','record_calls','inactive_at','updated_at_y','created_at_y','initial_contact_at',
                        'last_contact_at','next_activity_scheduled_at','start_business_hours_speed_to_lead_clock_at','discarded_by_id',
                        'discarded_at','original_prospect_id','merged_into_prospect_id','referral_reference_number',
                        'start_speed_to_lead_clock_at','secondary_lead_source_id','close_reason_details','referrer_id',
                        'close_reason_id', 'status_changed_at','created_at_x','updated_at_x', 'current_caregiver','original_sales_counselor_id','expected_move_timing_id'], axis = 1)

df_temp['lead_source_id'].fillna(0,inplace=True)
df_temp['expected_stay_type'].fillna(5,inplace=True)

col_names = ['id', 'community_id', 'sales_counselor_id', 'stage_id', 'lead_source_id','score_id',
             'familiarity_with_community','account_id_x','account_id_y','likelihood_to_move_from_home']

for col in col_names:
    df_temp[col] = df_temp[col].astype('category',copy=False)
    
combined_df = df_temp.merge(df2, how = 'left', left_on = 'community_id', right_on = 'community_id')
final_df = combined_df.dropna(subset=['occupancy_points'])
#final_df['created_at_x'] = pd.to_datetime(final_df['created_at_x'], errors = 'coerce')
#final_df['updated_at_x'] = pd.to_datetime(final_df['updated_at_x'], errors = 'coerce')
final_df1 = final_df.iloc[2200000:2400000,:]
final_df1 = final_df1.drop(['community_id_x','community_id_y'], axis = 1)
'''
Train-Test split
'''
names = ['id_x', 'community_id', 'sales_counselor_id', 'stage_id',
       'lead_source_id', 'status', 'score_id', 'familiarity_with_community',
       'likelihood_to_move_from_home', 'account_id_x', 'expected_stay_type',
       'account_id_y', 'occupancy_goal_percent', 'id_y', 'unit_id',
       'risk_level', 'stay_type', 'floor_plan_id',
       'care_type_id', 'vacant_status', 'beds', 'baths',
       'market_rate', 'second_resident_market_rate']
x_train, x_test, y_train, y_test = train_test_split(final_df1.iloc[:,:-1], final_df1.iloc[:,-1],
                                                    test_size=0.2, random_state=1)

rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=0, n_jobs= -1)
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)
r2_rf = rf.score(x_test, y_test)
mse_rf = mean_squared_error(y_test, rf_predict)
mae_rf = median_absolute_error(y_test, rf_predict)
importance_rf = rf.feature_importances_
j1 = range(len(importance_rf))
index_val = [ i for (i,j) in zip(j1,importance_rf) if j >= 0.03 ]
j2 = [ j for (i,j) in zip(j1,importance_rf) if j >= 0.03 ]
names1 = [names[i] for i in index_val]
#j2 = [i for i in importance_rf if i >= 0.03]
# plot feature importance
plt.bar([x for x in range(len(j2))], j2)
plt.title('Feature importance plot based on Random Forest regressor')
plt.xlabel('Feature name')
plt.xticks(range(len(j2)), names1, rotation=70)
plt.ylabel('Feature importance value')
plt.show()


lin_reg = LinearRegression().fit(x_train, y_train)
lin_predict = lin_reg.predict(x_test)
r2_linreg = lin_reg.score(x_test, y_test)
mse_linreg = mean_squared_error(y_test, lin_predict)
mae_linreg = median_absolute_error(y_test, lin_predict)
importance_linreg = lin_reg.coef_
# plot feature importance
j1 = range(len(importance_linreg))
index_val = [ i for (i,j) in zip(j1,importance_linreg) if j >= 0.01 ]
j2 = [ j for (i,j) in zip(j1,importance_linreg) if j >= 0.01 ]
names1 = [names[i] for i in index_val]
# plot feature importance
plt.bar([x for x in range(len(j2))], j2)
plt.title('Feature importance plot based on Linear regressor')
plt.xlabel('Feature name')
plt.xticks(range(len(j2)), names1, rotation=70)
plt.ylabel('Feature importance value')
plt.show()