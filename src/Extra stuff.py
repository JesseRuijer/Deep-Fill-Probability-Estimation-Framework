#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:48:14 2026

@author: jesseruijer
"""

#To print
#print(df_Event.iloc[0:10, 0:5])
#Keys
#print(mat_data_MO.keys())
#Find something specific print(df_Event[df_Event[0] == 24581757])

# print(df_Event.head())
# print(df_Event.tail())

# describe gives some nice stats stuff on dfs 
#print(df_Event_without_noise.describe())


#Search in DF
# print(df_Event_without_noise.iloc[15:25,0:7])
# print(df_MO_without_noise.iloc[0:5,0:9])

#How to import from other scripts
#from Functions import time_in_hours
#print(time_in_hours(3600000))


# print(order_life(32398125, cleandata))

#Below shows at end of day spams 68s to cancel outstanding orders in full 
# mask3 = (
#     (df_Event["Type"] == 68) & 
#     (df_Event["TOD"] > 55800000) & 
#     (df_Event["TOD"] <= 57600000)
#     )


#When you add the .values it changes structure from pandas df to a numpy array much faster for calculations
# testdf = pd.DataFrame({
#     '0': [1,2,3],
#     '1':[2,3,4]    
#     }
#     )

# print(testdf)
# print(testdf.values)

#Proving the Vol Ahead works 
# print(regressormatrix.head())
# print(cleandata['Event'].loc[91742])
# print(cleandata['BuyVol'].loc[91742])
# print(cleandata['BuyPrice'].loc[91742][0])


#Below prints how many 1s and 0s we had
# print("\n Total Fills vs Cancels")
# print(regressormatrix["Fill_NoFill"].value_counts())

#Below prints how many counts of Types we had
# print(rawdata["Event"]["Type"].value_counts())


#This was for p values but i dont think they are relevant when are sample size is this big, since i think standard error pretty much goes to zero when n is this big in our case couple hundred thousand 

# #Use statsmodeling library to present statistical evidence for findings p values etc, since technically this doesnt work or is very hard with sci kit learn i think
# #For statsmodeling lib you have to manually enter a col of 1s for intercept
# X_train_sm = sm.add_constant(X_train_standardised)
# #Creating statistical model logistic regression
# stat_model = sm.Logit(y_train.values, X_train_sm)
# #Fitting the model
# result = stat_model.fit(disp=False)
# #Build a pandas df to present result
# stats_df = pd.DataFrame({
#     "Feature": log_mdl_features,
#     "Coefficient (Log Odds)": result.params[1:],  # write [1:] to slice the array removing the first part since that was the artificial col of 1s we had to manually add before in order for statsmodel lib to work
#     "Odds Ratio": np.exp(result.params[1:]),
#     "P-Value": result.pvalues[1:]
# })

# print(stats_df)

















