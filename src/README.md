#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:54:19 2026

@author: jesseruijer
"""

'''

**Deep Fill Probability Prediction Engine Framework**

Coding framework that given level 3 LOB data employs various ML models to predict fill probabilities. This repository contains all the code used in this project.

**Project Structure**

**How to use**
The programme requires LOB level 3 tick data in the following format

## Data Prerequisites & Structure

Due to licensing and size constraints, the raw Level 3 NASDAQ Limit Order Book data is not included in this repository. To run this pipeline, you must provide your own MATLAB (`.mat`) data files formatted to match the expected schema.

### 1. Folder Structure & Naming Conventions
The automated file manager (`FileManager.py`) expects raw data to be organized by ticker and strictly named using the `TICKER_YYYYMMDD` format. 

Your `data/` directory must look exactly like this:

```text
repository_root/
│
├── data/
│   ├── raw/
│   │   └── INTC_NASDAQ/
│   │       ├── INTC_20140708_NASDAQ.mat      # Main LOB data
│   │       └── MO/
│   │           └── INTC_20140708.mat         # Market Order data
│   │
│   └── processed/                            # Parquet files will generate here


**Requirements**

pandas
numpy
pytorch
scipy
config
gc
matplotlib
seaborn
os
torch
shap
sklearn
joblib
pathlib
optuna
lightgbm
random
tkinter
json
glob
sys

If you are seriously interested in reviewing/using the code, you can reach out to me on jesse.ruijer@gmail.com to discuss the code with me.

'''
