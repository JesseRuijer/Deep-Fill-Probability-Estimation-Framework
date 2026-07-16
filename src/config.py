#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:04:48 2026

@author: jesseruijer
"""






#This below is for use in Main

CURRENT_LR_MODEL = 'Logistic_Regression_Models_V1.joblib'
CURRENT_LGBM_MODEL = 'LGBM_Models_V1.joblib'
CURRENT_FNN_MODEL_WEIGHTS = 'FFN_Models_V1_Weights.pth'
CURRENT_FNN_MODEL_METADATA = 'FFN_Models_V1_Metadata.joblib' #Metadata is just a cool name for data about data hihi

#This below is for use in userscript

USER_LR_MODEL = 'Logistic_Regression_Models_user.joblib'
USER_LGBM_MODEL = 'LGBM_Models_user.joblib'
USER_FNN_MODEL_WEIGHTS = 'FFN_Models_user_Weights.pth'
USER_FNN_MODEL_METADATA = 'FFN_Models_user_Metadata.joblib' #Metadata is just a cool name for data about data hihi

#put this on true when not including full training day
#also when changing this you have to delete and resave in main 
DONT_INCLUDE_FULL_TRAINING_DAY = False 

#just has some project wide constans 

TARGET = 'FillNoFill'
TICK = 'INTC'

#These constants below are based on the ExploratoryDataScript results atm, but in the future could write something to extract it immediatley  

HEARTBEAT_INTERVAL = 1000
MAX_HEARTBEATS = 50 #Since exponential decay in fill prob as order been placed for longer, and to save RAM, need this implementation to only have a max of 50 heartbeats for a single order ID, but if i have more computing power, then this could be increased

LOOKBACK_WINDOW = 3000

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
     
    # 1. Base Market & Order Status
    #'Regime',
    'TimeTillMarketClose',
    #'IsFinalMinute',
   # 'BidSize',
    #'AskSize',
    'TotalQueueSize',


    # 2. Spreads & Imbalances
    'BASpread',
    #'QImbalance',
    'AbsQImbalance',
    #'TotalVolImbalance',
    #'WeightedVolImbalance',
    #'MicroMidDeviation',
    #'CancelationRatio',
    #'OrderFlowImbalance',

    # 3. Event-Time Deltas (Tick-by-Tick Changes)
    #'EventDeltaMidprice',
    #'EventMicroMidDeviation',
    #'EventDeltaMicroprice',
    #'EventDeltaDistanceToTouch',
    #'EventDeltaOrderFlowImbalance',

    # 4. Rolling Moments (Event-Time Volatility & Extremes)

    #'RollingStd_OrderFlowImbalance',
    #'RollingStd_BASpread',
    'RollingStd_QImbalance',
    #'RollingStd_WeightedVolImbalance',
    #'RollingStd_MicroMidDeviation',
    #'RollingSkew_OrderFlowImbalance',
    #'RollingMax_OrderFlowImbalance',
    #'RollingMin_OrderFlowImbalance',
    #'RollingMax_BASpread',
    #'RollingMin_BASpread',


    # 5. Trailing Volumes & Lookbacks
    #'LookBackHiddenVol',
    #'MOTrailingVolBuy',
    #'MOTrailingVolSell',
    #'MOTrailingVolRatio',
    'LOTrailingVolPlaced',
    #'LOTrailingVolCanceled',
    #'LOTrailingVolExecuted',
    #'LOTrailingPlaceCancelRatio',
    #'LOTrailingPlaceExecuteRatio',

    # 6. Dynamic Order Metrics (Heartbeat Engine)
    'DistanceToTouch',
    'DistanceToMicroprice',
    'LogVolAhead',
    #'QueuePositionRatio',
    'TimeSincePlacement',
    'Is_Initial_Placement',


    # 7. Clock-Time Deltas (The 1-Second Lookbacks)
    #'ClockDeltaMidprice',
    #'ClockMicroMidDeviation',
    #'ClockDeltaDistanceToMicroprice',
    #'ClockDeltaDistanceToTouch',
    #'ClockDeltaLogVolAhead',
    #'ClockDeltaOrderFlowImbalance',

    # 8. Speed Metrics
    #'Speed_DeltaMidprice',
    #'Speed_DeltaDistanceToMicroprice',
    #'Speed_DeltaDistanceToTouch',
    #'Speed_MicroMidDeviation',
    #'Speed_OrderFlowImbalance',


    # 9. Market Order (MO) & Sweep Impacts
    'TimeSinceLastMO',
    #'MOCount10ms',
    #'SweepNoSweep',
    #'SweepInLast_2000ms',
    #'SweepIntensity_2000ms'
]

FNN_MODEL_FEATURES = [
     
    # 1. Base Market & Order Status
    'Regime',
    'TimeTillMarketClose',
    'IsFinalMinute',
    'BidSize',
    'AskSize',
    'TotalQueueSize',

    # 2. Spreads & Imbalances
    'BASpread',
    'QImbalance',
    'AbsQImbalance',
    'TotalVolImbalance',
    'WeightedVolImbalance',
    'OrderFlowImbalance',

    # 3. Event-Time Deltas (Tick-by-Tick Changes)
    'EventDeltaMidprice',
    'EventDeltaMicroprice',
    'EventDeltaDistanceToTouch',
    'EventDeltaOrderFlowImbalance',

    # 4. Rolling Moments (Event-Time Volatility & Extremes)
    'RollingStd_Microprice',
    'RollingStd_OrderFlowImbalance',
    'RollingStd_BASpread',
    'RollingStd_QImbalance',
    'RollingStd_WeightedVolImbalance',
    'RollingSkew_OrderFlowImbalance',
    'RollingMax_OrderFlowImbalance',
    'RollingMin_OrderFlowImbalance',
    'RollingMax_BASpread',
    'RollingMin_BASpread',
    'RollingMax_Microprice',
    'RollingMin_Microprice',

    # 5. Trailing Volumes & Lookbacks
    'MOTrailingVolBuy',
    'MOTrailingVolSell',
    'MOTrailingVolRatio',
    'LOTrailingVolPlaced',
    'LOTrailingVolCanceled',
    'LOTrailingVolExecuted',
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',

    # 6. Dynamic Order Metrics (Heartbeat Engine)
    'DistanceToTouch',
    'DistanceToMicroprice',
    'LogVolAhead',
    'QueuePositionRatio',
    'TimeSincePlacement',
    'Is_Initial_Placement',

    # 7. Clock-Time Deltas (The 1-Second Lookbacks)
    'ClockDeltaMidprice',
    'ClockDeltaDistanceToMicroprice',
    'ClockDeltaDistanceToTouch',
    'ClockDeltaLogVolAhead',
    'ClockQImbalance',
    'ClockDeltaOrderFlowImbalance',

    # 8. Speed Metrics
    'Speed_DeltaMidprice',
    'Speed_DeltaDistanceToMicroprice',
    'Speed_DeltaDistanceToTouch',
    'Speed_DeltaOrderFlowImbalance',

    # 9. Market Order (MO) & Sweep Impacts
    'TimeSinceLastMO',
    'MOCount10ms',
    'SweepNoSweep',
    'SweepInLast_2000ms',
    'SweepIntensity_2000ms'
    
    ]

#Just for using the SHAP function, i fully put all the relevant featues in here
LGBM_MODEL_FEATURES = [
     
       
     # 1. Base Market & Order Status
     #'Regime',
     'TimeTillMarketClose',
     #'IsFinalMinute',
     #'BidSize',
     #'AskSize',
     'TotalQueueSize',


     # 2. Spreads & Imbalances
     #'BASpread',
     #'QImbalance',
     'AbsQImbalance',
     'TotalVolImbalance',
     'WeightedVolImbalance',
     #'MicroMidDeviation',
     #'CancelationRatio',
     #'OrderFlowImbalance',

     # 3. Event-Time Deltas (Tick-by-Tick Changes)
     #'EventDeltaMidprice',
     #'EventMicroMidDeviation',
     'EventDeltaMicroprice',
     #'EventDeltaDistanceToTouch',
     'EventDeltaOrderFlowImbalance',

     # 4. Rolling Moments (Event-Time Volatility & Extremes)

     'RollingStd_OrderFlowImbalance',
     #'RollingStd_BASpread',
     #'RollingStd_QImbalance',
     #'RollingStd_WeightedVolImbalance',
     #'RollingStd_MicroMidDeviation',
     #'RollingSkew_OrderFlowImbalance',
     'RollingMax_OrderFlowImbalance',
     #'RollingMin_OrderFlowImbalance',
     #'RollingMax_BASpread',
     #'RollingMin_BASpread',


     # 5. Trailing Volumes & Lookbacks
     #'LookBackHiddenVol',
     #'MOTrailingVolBuy',
     'MOTrailingVolSell',
     'MOTrailingVolRatio',
     'LOTrailingVolPlaced',
     'LOTrailingVolCanceled',
     #'LOTrailingVolExecuted',
     'LOTrailingPlaceCancelRatio',
     #'LOTrailingPlaceExecuteRatio',

     # 6. Dynamic Order Metrics (Heartbeat Engine)
     'DistanceToTouch',
     'DistanceToMicroprice',
     'LogVolAhead',
     'QueuePositionRatio',
     'TimeSincePlacement',
     'Is_Initial_Placement',


     # 7. Clock-Time Deltas (The 1-Second Lookbacks)
     #'ClockDeltaMidprice',
     #'ClockMicroMidDeviation',
     #'ClockDeltaDistanceToMicroprice',
     #'ClockDeltaDistanceToTouch',
     'ClockDeltaLogVolAhead',
     #'ClockDeltaOrderFlowImbalance',

     # 8. Speed Metrics
     #'Speed_DeltaMidprice',
     'Speed_DeltaDistanceToMicroprice',
     #'Speed_DeltaDistanceToTouch',
     #'Speed_MicroMidDeviation',
     #'Speed_OrderFlowImbalance',


     # 9. Market Order (MO) & Sweep Impacts
     'TimeSinceLastMO',
     #'MOCount10ms',
     #'SweepNoSweep',
     #'SweepInLast_2000ms',
     #'SweepIntensity_2000ms'
]

#Note not all feautures here are varaibles in the model but its just useful to have them in a list like this
UNIVERSAL_FEATURES = [ # Features that apply to any LO, so for ex midprice yes, but DistanceToMidprice no since thats specific per LO, and also that has been calculated during start of script and not all the way at end
    'TOD',
                      
    # Base Market
    'Regime',
    'TimeTillMarketClose',
    'IsFinalMinute',
    'BidSize',
    'AskSize',
    'BestBid',
    'BestAsk',
    'Midprice',
    'Microprice',
    'TotalQueueSize',

    # Spreads & Imbalances
    'BASpread',
    'QImbalance',
    'AbsQImbalance',
    'TotalVolImbalance',
    'WeightedVolImbalance',
    #'MicroMidDeviation',
    'CancelationRatio', #cant use this in a model since it gives lookaheadbias
    'OrderFlowImbalance',

    # Event-Time Deltas (Market-wide)
    'EventDeltaMidprice',
    'EventDeltaMicroprice',
    #'EventMicroMidDeviation',
    'EventDeltaOrderFlowImbalance',
    'EventDeltaBestBid',
    'EventDeltaBestAsk',

    # Rolling Moments
    'RollingStd_Microprice',
    'RollingStd_OrderFlowImbalance',
    'RollingStd_BASpread',
    'RollingStd_QImbalance',
    'RollingStd_WeightedVolImbalance',
    #'RollingStd_MicroMidDeviation',
    'RollingSkew_OrderFlowImbalance',
    'RollingMax_OrderFlowImbalance',
    'RollingMin_OrderFlowImbalance',
    'RollingMax_BASpread',
    'RollingMin_BASpread',
    'RollingMax_Microprice',
    'RollingMin_Microprice',

    # Trailing Volumes & Lookbacks
    'LookBackHiddenVol',
    'MOTrailingVolBuy',
    'MOTrailingVolSell',
    'MOTrailingVolRatio',
    'LOTrailingVolPlaced',
    'LOTrailingVolCanceled',
    'LOTrailingVolExecuted',
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',

]

DYNAMIC_FEATURES = [ #Features that change (in general) depending on the LO you look at
    # Static Order Attributes

    
    # Event-Time Deltas (Order-Direction Dependent)
    'EventDeltaDistanceToMicroprice',
    'EventDeltaDistanceToTouch',

    # Heartbeat Engine Recalculations
    'DistanceToTouch',
    'DistanceToMicroprice',
    'VolAhead',
    'LogVolAhead',
    'QueuePositionRatio',
    'TimeSincePlacement',
    
         
     #Before final submission train one more time also including this below:
     #'Is_Initial_Placement',


    # Clock-Time Deltas (Order-Direction/Position Dependent)
    'ClockDeltaDistanceToMicroprice',
    'ClockDeltaDistanceToTouch',
    'ClockDeltaLogVolAhead',

    # Speed Metrics (Order-Direction/Position Dependent)
    'Speed_DeltaDistanceToMicroprice',
    'Speed_DeltaDistanceToTouch',
    'Speed_DistanceToMicroprice'
]


ALL_FEATURES = [
    
    #This is not necessarily everything the full matrix returns sinec some stuff i needed it to return to use later but thats definitely not a feature
    
    # 1. Base Market & Order Status
    'TOD',
    'Regime',
    'TimeTillMarketClose',
    'IsFinalMinute',
    'BidSize',
    'AskSize',
    'Midprice',
    'Microprice',
    'TotalQueueSize',

    # 2. Spreads & Imbalances
    'BASpread',
    'QImbalance',
    'AbsQImbalance',
    'TotalVolImbalance',
    'WeightedVolImbalance',
    #'MicroMidDeviation',
    'CancelationRatio',#cant use this in a model since it gives lookaheadbias
    'OrderFlowImbalance',

    # 3. Event-Time Deltas (Tick-by-Tick Changes)
    'EventDeltaMidprice',
    #'EventMicroMidDeviation',
    'EventDeltaMicroprice',
    'EventDeltaDistanceToTouch',
    'EventDeltaOrderFlowImbalance',

    # 4. Rolling Moments (Event-Time Volatility & Extremes)
    'RollingStd_Microprice',
    'RollingStd_OrderFlowImbalance',
    'RollingStd_BASpread',
    'RollingStd_QImbalance',
    'RollingStd_WeightedVolImbalance',
   # 'RollingStd_MicroMidDeviation',
    'RollingSkew_OrderFlowImbalance',
    'RollingMax_OrderFlowImbalance',
    'RollingMin_OrderFlowImbalance',
    'RollingMax_BASpread',
    'RollingMin_BASpread',
    'RollingMax_Microprice',
    'RollingMin_Microprice',

    # 5. Trailing Volumes & Lookbacks
    'LookBackHiddenVol',
    'MOTrailingVolBuy',
    'MOTrailingVolSell',
    'MOTrailingVolRatio',
    'LOTrailingVolPlaced',
    'LOTrailingVolCanceled',
    'LOTrailingVolExecuted',
    'LOTrailingPlaceCancelRatio',
    'LOTrailingPlaceExecuteRatio',

    # 6. Dynamic Order Metrics (Heartbeat Engine)
    'DistanceToTouch',
    'DistanceToMicroprice',
    'VolAhead',
    'LogVolAhead',
    'QueuePositionRatio',
    'TimeSincePlacement',
    
         
     #Before final submission train one more time also including this below:
     #'Is_Initial_Placement',


    # 7. Clock-Time Deltas (The 1-Second Lookbacks)
    'ClockDeltaMidprice',
    #'ClockMicroMidDeviation',
    'ClockDeltaDistanceToMicroprice',
    'ClockDeltaDistanceToTouch',
    'ClockDeltaLogVolAhead',
    'ClockQImbalance',
    'ClockDeltaOrderFlowImbalance',

    # 8. Speed Metrics
    'Speed_DeltaMidprice',
    'Speed_DeltaDistanceToMicroprice',
    'Speed_DeltaDistanceToTouch',
    #'Speed_MicroMidDeviation',
    'Speed_OrderFlowImbalance',
    'Speed_DistanceToMicroprice',

    # 9. Market Order (MO) & Sweep Impacts
    'TimeSinceLastMO',
    'MOCount10ms',
    'SweepNoSweep',
    'SweepInLast_2000ms',
    'SweepIntensity_2000ms'
]







