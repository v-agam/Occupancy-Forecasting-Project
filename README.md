# Occupancy-Forecasting-Project

This project was taken up as a part of my MS in Analytics coursework where I created the complete solution from scratch for a start-up company which provides senior living customer relationship management platform.
To help in managing the fast growing and increasingly complex data, and utilizing it for creating business value, data analytics based solutions are being used by the company. Occupancy is the biggest driver
of profitability in the senior living industry, where margins can be very thin. The occupancy figures play a vital role in strategy planning but often have low visibility into any incoming trend. Forecasting
the occupancy figures creates opportunity to mitigate a possible decline in the occupancy level. Forecasting of occupancy percent, move-ins and move-outs at a community level is one solution
which is implemented in this project work.

Due to confidentiality of the data, it is not shown here. The basic operations performed are as below:

To obtain a basic understanding of the raw data, EDA is performed on the data which consisted of 34 different tables and a total of approximately 260 features. A preliminary analysis of raw data indicated
that only few of the tables were relevant for the purpose of occupancy forecasting. Further exploration of data showed that some tables have large number of values in them. Example: "prospects" table
contains 920,056 rows and 39 columns, while "housing contracts" table consists of 69710 rows and 45 columns. Merging of these big tables with others consumes large amount of processor memory which is
often difficult to obtain on a typical PC/laptop. Thus, it necessitated the need to reduce dimensionality of the data and remove rows and columns based on different criteria (explained in detail in section 4.1)
before performing the merge on tables. A total of 12 tables are shortlisted and then merged together to form a single data source of approximately 80,000 rows and 100 columns.

Plots for data and density distribution are used to understand the input data. It also helps in identifying possible outliers in the data. A pairwise scatter plot
is also used to get a bird's eye view of the relationships between the features of a given table.

These tasks are followed by data preprocessing which included steps of data cleaning and imputation of missing values, categorical feature conversion, data standardization, correlation removal
and conversion to time-series data.

This is followed by feature selection and time series analysis and forecasting. The mean squared error metric has been taken for model performance evaluation and selection.
