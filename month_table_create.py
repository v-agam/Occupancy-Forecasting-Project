# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 21:15:27 2021

@author: Legion
"""
import calendar
import pandas as pd
from datetime import datetime

def create_month_table(month, year):
    """
    Create shell table
    """
    months = []
    years = []
    last_days_of_month = []

    #fill in months and years
    for i in range(50):
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

