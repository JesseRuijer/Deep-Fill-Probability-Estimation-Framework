#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:04:48 2026

@author: jesseruijer
"""

#just has some project wide constans 

MARKET_OPEN_TIME = 34200000 # 9:30 AM
MARKET_CLOSE_TIME = 57600000 # 4 PM
MARKET_CLOSE_TIME_INCLUDING_CANC_SPAM = 57660000 # 4:01 PM

SOMARKET_NOISE = 36000000 # 10 AM
EOMARKET_NOISE =  55800000 # 3:30 PM



LOGISTIC_MODEL_FEATURES = ['AbsQImbalance', 'Weighted Vol Imbalance', 
              "DistanceToTouch", 'LogVolAhead', "LookBackHiddenVol", 'TimeSincePlacement', 'DistanceToMidprice'] 

LGBM_MODEL_FEATURES = ['AbsQImbalance', 'Weighted Vol Imbalance', 
              "DistanceToTouch", 'LogVolAhead', "LookBackHiddenVol" ,'BASpread', 'QImbalance', 'TotalVolImbalance', 'Midprice', 'Microprice', 
                                        'MOTrailingVol100ms', 'MOTrailingOrders100ms', 'LOTrailingVolPlaced100ms', 'LOTrailingCountOrdersPlaced100ms', 
                                        'LOTrailingVolCanceled100ms', 'LOTrailingCountOrdersCanceled100ms', 'LOTrailingVolExecuted100ms',
                                        'LOTrailingCountOrdersExecuted100ms', 'VolAhead', 'TimeSincePlacement', 'DistanceToMidprice']

TARGET = 'FillNoFill'

HEARTBEAT_INTERVAL = 10000








