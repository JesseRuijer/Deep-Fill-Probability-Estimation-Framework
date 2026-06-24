#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:04:48 2026

@author: jesseruijer
"""

#just has some project wide constans 

TARGET = 'FillNoFill'

HEARTBEAT_INTERVAL = 10000

LOOKBACK_WINDOW = 100

EVENT_TIME_DELTA = 50 # look back 50 event ticks

FEATURE_DELTA = 1000

MARKET_OPEN_TIME = 34200000 # 9:30 AM
MARKET_CLOSE_TIME = 57600000 # 4 PM
MARKET_CLOSE_TIME_INCLUDING_CANC_SPAM = 57660000 # 4:01 PM

SOMARKET_NOISE = 36000000 # 10 AM
EOMARKET_NOISE =  55800000 # 3:30 PM



#CHANGE FEATURE SELECTION FOR MODELS STILL 

#For the features that get passed to the models i should ban all raw prices and absolute timestamps as that will cause models to perform poorly on new data

LOGISTIC_MODEL_FEATURES = [
    'AbsQImbalance', 
    'DeltaLogVolAhead',
    'DeltaDistanceToTouch',
    'DeltaMidprice',
    'DeltaDistanceToMidprice',
    'DistanceToTouch', 
    'LogVolAhead', 
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',
    'MicroMidDeviation',
    'MOTrailingVolRatio',
    'MOCount10ms',
    'RollingMidPrice',
    'OrderFlowImbalance',
    'QueuePositionRatio',
    'TimeSincePlacement', 
    'TimeTillMarketClose', 
    'TimeSinceLastMO',
    'WeightedVolImbalance'
]

#Just for using the SHAP function, i fully put all the relevant featues in here
LGBM_MODEL_FEATURES = [
    'AbsQImbalance', 
    'AskSize', 
    'BASpread', 
  #  'BestAsk', 
  #  'BestBid', 
    'BidSize', 
    'CancelationRatio', 
    'DeltaLogVolAhead',
    'DeltaDistanceToTouch',
    'DeltaMidprice',
    'DeltaDistanceToMidprice',
    'DistanceToMicroprice', 
    'DistanceToTouch', 
    #'InitialPlacementTime', 
    'IsFinalMinute', 
    'LogVolAhead', 
    'LookBackHiddenVol',
    'LOTrailingCountOrdersCanceled', 
    'LOTrailingCountOrdersExecuted', 
    'LOTrailingCountOrdersPlaced', 
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',
  #  'Microprice', 
   # 'Midprice', 
   'MicroMidDeviation',
    'MOTrailingOrders', 
    'MOTrailingVolRatio',
    'MOCount10ms',
    'OrderFlowImbalance',
    'QueuePositionRatio',
    'QImbalance', 
    'RollingMidPrice',
    'TimeSincePlacement', 
    'TimeTillMarketClose', 
    'TimeSinceLastMO',
   # 'TOD', 
    'TotalVolImbalance', 
   # 'VolAhead', 
    'WeightedVolImbalance'
]

UNIVERSAL_FEATURES = [          # Features that apply to any LO, so for ex midprice yes, but DistanceToMidprice no since thats specific per LO
    'AbsQImbalance', 
    'AskSize', 
    'BASpread', 
    'BestAsk', 
    'BestBid', 
    'BidSize', 
    'CancelationRatio', 
    'DeltaMidprice',
    'IsFinalMinute', 
    'LookBackHiddenVol',
    'LOTrailingCountOrdersCanceled', 
    'LOTrailingCountOrdersExecuted', 
    'LOTrailingCountOrdersPlaced', 
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',
    'Microprice', 
    'MicroMidDeviation',
    'Midprice', 
    'MOTrailingOrders', 
    'MOTrailingVolRatio',
    'OrderFlowImbalance',
    'QImbalance', 
    'Regime', 
    'RollingMidPrice',
    'TimeTillMarketClose', 
    'TOD', 
    'TotalVolImbalance', 
    'WeightedVolImbalance'
]

DYNAMIC_FEATURES = [    #Features that change (in general) depending on the LO you look at
    'DeltaLogVolAhead',
    'DeltaDistanceToTouch',
    'DeltaDistanceToMidprice',
    'DistanceToMicroprice', 
    'DistanceToTouch', 
    'InitialPlacementTime',
    'LogVolAhead', 
    'MOCount10ms',
    'QueuePositionRatio',
    'TimeSincePlacement', 
    'TimeSinceLastMO',
    'VolAhead'
    ]

#Just commented some of the features below out since was finding best features for logistic and that cant take stationary features well like best bid for ex

ALL_FEATURES = [
    'AbsQImbalance', 
    'AskSize', 
    'BASpread', 
    #'BestAsk', 
    #'BestBid', 
    'BidSize', 
    'CancelationRatio', 
    'DeltaLogVolAhead',
    'DeltaMidprice',
    'DeltaDistanceToTouch',
    'DeltaDistanceToMidprice',
    'DistanceToMicroprice', 
    'DistanceToTouch', 
  #  'InitialPlacementTime', 
    'IsFinalMinute', 
    'LogVolAhead', 
    'LookBackHiddenVol',
    'LOTrailingCountOrdersCanceled', 
    'LOTrailingCountOrdersExecuted', 
    'LOTrailingCountOrdersPlaced', 
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',
    #'Microprice', 
    #'Midprice', 
    'MicroMidDeviation',
    'MOTrailingOrders', 
    'MOTrailingVolRatio',
    'MOCount10ms',
    'OrderFlowImbalance',
    'QueuePositionRatio',
    'QImbalance', 
    'Regime',
    'RollingMidPrice',
    'TimeSincePlacement', 
    'TimeTillMarketClose', 
    'TimeSinceLastMO',
   # 'TOD', 
    'TotalVolImbalance', 
    #'VolAhead', 
    'WeightedVolImbalance'
]








