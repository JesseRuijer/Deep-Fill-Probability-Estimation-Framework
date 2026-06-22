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

FEATURE_DELTA = 1000

MARKET_OPEN_TIME = 34200000 # 9:30 AM
MARKET_CLOSE_TIME = 57600000 # 4 PM
MARKET_CLOSE_TIME_INCLUDING_CANC_SPAM = 57660000 # 4:01 PM

SOMARKET_NOISE = 36000000 # 10 AM
EOMARKET_NOISE =  55800000 # 3:30 PM



#CHANGE FEATURE SELECTION FOR MODELS STILL 

LOGISTIC_MODEL_FEATURES = [
    'AbsQImbalance', 
    'DeltaLogVolAhead',
    'DeltaDistanceToTouch',
    'DistanceToTouch', 
    'LogVolAhead', 
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',
    'MOTrailingVolRatio',
    'TimeSincePlacement', 
    'TimeTillMarketClose', 
    'WeightedVolImbalance'
]

LGBM_MODEL_FEATURES = [
    'AbsQImbalance', 
    'DeltaLogVolAhead',
    'DeltaDistanceToTouch',
    'DistanceToTouch', 
    'LogVolAhead', 
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',
    'Microprice', 
    'MOTrailingVolRatio',
    'TimeSincePlacement', 
    'TotalVolImbalance', 
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
    'IsFinalMinute', 
    'LookBackHiddenVol',
    'LOTrailingCountOrdersCanceled', 
    'LOTrailingCountOrdersExecuted', 
    'LOTrailingCountOrdersPlaced', 
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',
    'Microprice', 
    'Midprice', 
    'MOTrailingOrders', 
    'MOTrailingVolRatio',
    'QImbalance', 
    'Regime', 
    'TimeTillMarketClose', 
    'TOD', 
    'TotalVolImbalance', 
    'WeightedVolImbalance'
]

DYNAMIC_FEATURES = [    #Features that change (in general) depending on the LO you look at
    'DeltaLogVolAhead',
    'DeltaDistanceToTouch',
    'DistanceToMidprice', 
    'DistanceToTouch', 
    'InitialPlacementTime',
    'LogVolAhead', 
    'TimeSincePlacement', 
    'VolAhead'
    ]

ALL_FEATURES = [
    'AbsQImbalance', 
    'AskSize', 
    'BASpread', 
    'BestAsk', 
    'BestBid', 
    'BidSize', 
    'CancelationRatio', 
    'DeltaLogVolAhead',
    'DeltaDistanceToTouch',
    'DistanceToMidprice', 
    'DistanceToTouch', 
    'InitialPlacementTime', 
    'IsFinalMinute', 
    'LogVolAhead', 
    'LookBackHiddenVol',
    'LOTrailingCountOrdersCanceled', 
    'LOTrailingCountOrdersExecuted', 
    'LOTrailingCountOrdersPlaced', 
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',
    'Microprice', 
    'Midprice', 
    'MOTrailingOrders', 
    'MOTrailingVolRatio',
    'QImbalance', 
    'TimeSincePlacement', 
    'TimeTillMarketClose', 
    'TOD', 
    'TotalVolImbalance', 
    'VolAhead', 
    'WeightedVolImbalance'
]








