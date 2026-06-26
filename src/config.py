#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:04:48 2026

@author: jesseruijer
"""

#just has some project wide constans 

TARGET = 'FillNoFill'


#These constants below are based on the ExploratoryDataScript results atm, but in the future could write something to extract it immediatley  

HEARTBEAT_INTERVAL = 1000
MAX_HEARTBEATS = 50 #Since exponential decay in fill prob as order been placed for longer, and to save RAM, need this implementation to only have a max of 60 heartbeats for a single order ID, but if i have more computing power, then this could be increased

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
    'Vol',
    'Regime',
    'TimeTillMarketClose',
    'IsFinalMinute',
    'BidSize',
    'AskSize',
    'Midprice',
    'Microprice',

    # 2. Spreads & Imbalances
    'BASpread',
    'QImbalance',
    'AbsQImbalance',
    'TotalVolImbalance',
    'WeightedVolImbalance',
    'MicroMidDeviation',
    'CancelationRatio',

    # 3. Event-Time Deltas (Tick-by-Tick Changes)
    'EventDeltaMidprice',
    'EventMicroMidDeviation',
    'EventDeltaDistanceToMicroprice',
    'EventDeltaDistanceToTouch',
    'EventOrderFlowImbalance',

    # 4. Rolling Moments (Event-Time Volatility & Extremes)
    'RollingStd_Microprice',
    'RollingStd_EventOrderFlowImbalance',
    'RollingStd_BASpread',
    'RollingStd_QImbalance',
    'RollingStd_WeightedVolImbalance',
    'RollingStd_MicroMidDeviation',
    'RollingSkew_EventOrderFlowImbalance',
    'RollingMax_EventOrderFlowImbalance',
    'RollingMin_EventOrderFlowImbalance',
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
    'LogVolAhead',
    'QueuePositionRatio',
    'TimeSincePlacement',

    # 7. Clock-Time Deltas (The 1-Second Lookbacks)
    'ClockDeltaMidprice',
    'ClockMicroMidDeviation',
    'ClockDeltaDistanceToMicroprice',
    'ClockDeltaDistanceToTouch',
    'ClockDeltaLogVolAhead',
    'ClockQImbalance',
    'ClockOrderFlowImbalance',

    # 8. Speed Metrics
    'Speed_DeltaMidprice',
    'Speed_DeltaDistanceToMicroprice',
    'Speed_DeltaDistanceToTouch',
    'Speed_MicroMidDeviation',
    'Speed_QImbalance',
    'Speed_OrderFlowImbalance',
    'Speed_LogVolAhead',

    # 9. Market Order (MO) & Sweep Impacts
    'TimeSinceLastMO',
    'MOCount10ms',
    'TimeSinceLastSweep',
]

#Just for using the SHAP function, i fully put all the relevant featues in here
LGBM_MODEL_FEATURES = [
     
    # 1. Base Market & Order Status
    'Vol',
    'Regime',
    'TimeTillMarketClose',
    'IsFinalMinute',
    'BidSize',
    'AskSize',
    'Midprice',
    'Microprice',

    # 2. Spreads & Imbalances
    'BASpread',
    'QImbalance',
    'AbsQImbalance',
    'TotalVolImbalance',
    'WeightedVolImbalance',
    'MicroMidDeviation',
    'CancelationRatio',

    # 3. Event-Time Deltas (Tick-by-Tick Changes)
    'EventDeltaMidprice',
    'EventMicroMidDeviation',
    'EventDeltaDistanceToMicroprice',
    'EventDeltaDistanceToTouch',
    'EventOrderFlowImbalance',

    # 4. Rolling Moments (Event-Time Volatility & Extremes)
    'RollingStd_Microprice',
    'RollingStd_EventOrderFlowImbalance',
    'RollingStd_BASpread',
    'RollingStd_QImbalance',
    'RollingStd_WeightedVolImbalance',
    'RollingStd_MicroMidDeviation',
    'RollingSkew_EventOrderFlowImbalance',
    'RollingMax_EventOrderFlowImbalance',
    'RollingMin_EventOrderFlowImbalance',
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
    'LogVolAhead',
    'QueuePositionRatio',
    'TimeSincePlacement',

    # 7. Clock-Time Deltas (The 1-Second Lookbacks)
    'ClockDeltaMidprice',
    'ClockMicroMidDeviation',
    'ClockDeltaDistanceToMicroprice',
    'ClockDeltaDistanceToTouch',
    'ClockDeltaLogVolAhead',
    'ClockQImbalance',
    'ClockOrderFlowImbalance',

    # 8. Speed Metrics
    'Speed_DeltaMidprice',
    'Speed_DeltaDistanceToMicroprice',
    'Speed_DeltaDistanceToTouch',
    'Speed_MicroMidDeviation',
    'Speed_QImbalance',
    'Speed_OrderFlowImbalance',
    'Speed_LogVolAhead',

    # 9. Market Order (MO) & Sweep Impacts
    'TimeSinceLastMO',
    'MOCount10ms',
    'TimeSinceLastSweep',
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

    # Spreads & Imbalances
    'BASpread',
    'QImbalance',
    'AbsQImbalance',
    'TotalVolImbalance',
    'WeightedVolImbalance',
    'MicroMidDeviation',
    'CancelationRatio',

    # Event-Time Deltas (Market-wide)
    'EventDeltaMidprice',
    'EventMicroMidDeviation',
    'EventOrderFlowImbalance',

    # Rolling Moments
    'RollingStd_Microprice',
    'RollingStd_EventOrderFlowImbalance',
    'RollingStd_BASpread',
    'RollingStd_QImbalance',
    'RollingStd_WeightedVolImbalance',
    'RollingStd_MicroMidDeviation',
    'RollingSkew_EventOrderFlowImbalance',
    'RollingMax_EventOrderFlowImbalance',
    'RollingMin_EventOrderFlowImbalance',
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

    # Clock-Time Deltas calculated in Zone 1
    'ClockDeltaMidprice',
    'ClockMicroMidDeviation'
]

DYNAMIC_FEATURES = [ #Features that change (in general) depending on the LO you look at
    # Static Order Attributes
    'Vol',
    
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

    # Clock-Time Deltas (Order-Direction/Position Dependent)
    'ClockDeltaDistanceToMicroprice',
    'ClockDeltaDistanceToTouch',
    'ClockDeltaLogVolAhead',

    # Speed Metrics (Order-Direction/Position Dependent)
    'Speed_DeltaDistanceToMicroprice',
    'Speed_DeltaDistanceToTouch',
    'Speed_DistanceToTouch',
    'Speed_LogVolAhead',
    'Speed_DistanceToMicroprice'
]


ALL_FEATURES = [
    
    # 1. Base Market & Order Status
    'Vol',
    'Regime',
    'TimeTillMarketClose',
    'IsFinalMinute',
    'BidSize',
    'AskSize',
    'Midprice',
    'Microprice',

    # 2. Spreads & Imbalances
    'BASpread',
    'QImbalance',
    'AbsQImbalance',
    'TotalVolImbalance',
    'WeightedVolImbalance',
    'MicroMidDeviation',
    'CancelationRatio',

    # 3. Event-Time Deltas (Tick-by-Tick Changes)
    'EventDeltaMidprice',
    'EventMicroMidDeviation',
    'EventDeltaDistanceToMicroprice',
    'EventDeltaDistanceToTouch',
    'EventOrderFlowImbalance',

    # 4. Rolling Moments (Event-Time Volatility & Extremes)
    'RollingStd_Microprice',
    'RollingStd_EventOrderFlowImbalance',
    'RollingStd_BASpread',
    'RollingStd_QImbalance',
    'RollingStd_WeightedVolImbalance',
    'RollingStd_MicroMidDeviation',
    'RollingSkew_EventOrderFlowImbalance',
    'RollingMax_EventOrderFlowImbalance',
    'RollingMin_EventOrderFlowImbalance',
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

    # 7. Clock-Time Deltas (The 1-Second Lookbacks)
    'ClockDeltaMidprice',
    'ClockMicroMidDeviation',
    'ClockDeltaDistanceToMicroprice',
    'ClockDeltaDistanceToTouch',
    'ClockDeltaLogVolAhead',
    'ClockQImbalance',
    'ClockOrderFlowImbalance',

    # 8. Speed Metrics
    'Speed_DeltaMidprice',
    'Speed_DeltaDistanceToMicroprice',
    'Speed_DeltaDistanceToTouch',
    'Speed_MicroMidDeviation',
    'Speed_QImbalance',
    'Speed_OrderFlowImbalance',
    'Speed_DistanceToTouch',
    'Speed_LogVolAhead',
    'Speed_DistanceToMicroprice',

    # 9. Market Order (MO) & Sweep Impacts
    'TimeSinceLastMO',
    'MOCount10ms',
    'TimeSinceLastSweep',
]







