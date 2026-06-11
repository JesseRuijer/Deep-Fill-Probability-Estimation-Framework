#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:04:48 2026

@author: jesseruijer
"""

#just has some project wide constans 

MARKET_OPEN_TIME = 34200000 # 9:30 AM
MARKET_CLOSE_TIME = 57600000 # 4 PM

TRAIN_FILE_PATH = '../data/raw/INTC_NASDAQ/INTC_20140401_NASDAQ.mat'
TRAIN_FILE_PATH_MO = '../data/raw/INTC_NASDAQ/Market Order/INTC_20140401.mat'

TEST_FILE_PATH = '../data/raw/INTC_NASDAQ/INTC_20140424_NASDAQ.mat'
TEST_FILE_PATH_MO = '../data/raw/INTC_NASDAQ/Market Order/INTC_20140424.mat'