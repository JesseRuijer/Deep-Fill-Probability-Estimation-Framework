#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:05:38 2026

@author: jesseruijer
"""

#This script just handles the data entries into the programme via finder instead of having to manually do this (avoids typos and reduces time)
#Also codes a cache_file so if wanting to rerun the same data multiple times the finder doesnt keep on opening
#Note majority of this code was AI generated and changed by me to fit as I have no expertise in software engineering related to File managing on pc etc

import tkinter as tk
from tkinter import filedialog
import os
import json

CACHE_FILE = ".last_paths.json"

def select_files_via_finder():
    
    #GUI for selecting the files via finder
    
    root = tk.Tk()
    root.withdraw() 
    
    print("Opening Finder for selecting Main data set")
    main_file = filedialog.askopenfilename(title="Select the MAIN .mat file", filetypes=[("MATLAB files", "*.mat")])
    if not main_file: return None, None
        
    print("Opening Finder for selecting MO data set")
    mo_file = filedialog.askopenfilename(title="Select the MO .mat file", filetypes=[("MATLAB files", "*.mat")])
    
    return main_file, mo_file

def get_data_paths():
    
    #If you want to use new data or remain with the old data
    
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            saved_paths = json.load(f)

        print(f"MAIN: {os.path.basename(saved_paths['main_path'])}")
        print(f"MO:   {os.path.basename(saved_paths['mo_path'])}")
        
        user_choice = input("Press ENTER to use these files, or type 'n' to pick a new one(s): ")
        
        if user_choice.strip().lower() != 'n':
            return saved_paths['main_path'], saved_paths['mo_path']

    main_path, mo_path = select_files_via_finder()
    
    if main_path and mo_path:
        with open(CACHE_FILE, 'w') as f:
            json.dump({'main_path': main_path, 'mo_path': mo_path}, f)
            
    return main_path, mo_path

def generate_dynamic_paths(main_file_path):
    
    #Splitting the main file into the train and testing for bin and multi class output var
    
    filename = os.path.basename(main_file_path)
    parts = filename.split('_')
    
    ticker = parts[0]
    date = parts[1]
    formatted_date = f"{date[:4]}_{date[4:6]}_{date[6:]}"
    
    binary_out = f"../data/processed/{ticker}_BINARY_{formatted_date}.parquet"
    multi_out = f"../data/processed/{ticker}_MULTI_{formatted_date}.parquet"

    return binary_out, multi_out

def get_ml_training_paths():
    """Opens Finder for Train and Test selection, sorts them, and caches for fast reruns."""
    ML_CACHE_FILE = ".ml_cache.json"
    
    # =========================================================================
    # 1. CHECK SHORT-TERM MEMORY (CACHE)
    # =========================================================================
    if os.path.exists(ML_CACHE_FILE):
        with open(ML_CACHE_FILE, 'r') as f:
            saved_paths = json.load(f)
            
        print("\n--- ML DATA CACHE FOUND ---")
        # .get() prevents errors if a slot is empty
        if saved_paths.get('train_bin'): print(f"TRAIN (Bin):   {os.path.basename(saved_paths['train_bin'])}")
        if saved_paths.get('train_multi'): print(f"TRAIN (Multi): {os.path.basename(saved_paths['train_multi'])}")
        if saved_paths.get('test_bin'): print(f"TEST (Bin):    {os.path.basename(saved_paths['test_bin'])}")
        if saved_paths.get('test_multi'): print(f"TEST (Multi):  {os.path.basename(saved_paths['test_multi'])}")
        print("---------------------------")
        
        user_choice = input("Press [ENTER] to reuse these datasets, or type 'n' to pick new ones: ")
        
        if user_choice.strip().lower() != 'n':
            return saved_paths

    # =========================================================================
    # 2. OPEN FINDER IF NO CACHE OR USER TYPED 'n'
    # =========================================================================
    root = tk.Tk()
    root.withdraw()
    init_dir = os.path.abspath("../data/processed/")
    
    print("\n--- SELECT DATA FOR MACHINE LEARNING ---")
    
    print("Highlight the TRAINING file(s) you want to use (Hold Cmd/Shift for multiple).")
    train_files = filedialog.askopenfilenames(initialdir=init_dir, title="Select TRAIN Data", filetypes=[("All files", "*.*")])
    
    if not train_files:
        return {} # Returns empty if canceled

    print("Highlight the TESTING file(s) you want to use (Hold Cmd/Shift for multiple).")
    test_files = filedialog.askopenfilenames(initialdir=init_dir, title="Select TEST Data", filetypes=[("All files", "*.*")])
    
    if not test_files:
        return {} # Returns empty if canceled

    # Set up empty slots
    paths = {
        'train_bin': None, 'test_bin': None, 
        'train_multi': None, 'test_multi': None
    }

    # Automatically sort the files you highlighted into the right slots
    for f in train_files:
        if "BINARY" in f.upper(): paths['train_bin'] = f
        if "MULTI" in f.upper(): paths['train_multi'] = f

    for f in test_files:
        if "BINARY" in f.upper(): paths['test_bin'] = f
        if "MULTI" in f.upper(): paths['test_multi'] = f

    # =========================================================================
    # 3. SAVE CHOICES TO CACHE FOR NEXT TIME
    # =========================================================================
    with open(ML_CACHE_FILE, 'w') as f:
        json.dump(paths, f)

    return paths

########Add a funciton to select multiple files for main and mo 
def get_batch_data_paths():
    """Allows selecting multiple MAIN files and auto-locates their MO files."""
    root = tk.Tk()
    root.withdraw() 
    
    print("\n--- BATCH DATA INGESTION ---")
    print("Opening Finder... Highlight as many MAIN .mat files as you want.")
    
    main_files = filedialog.askopenfilenames(
        title="Select MAIN .mat files (Hold Cmd/Shift for multiple)", 
        filetypes=[("MATLAB files", "*.mat")]
    )
    
    if not main_files: 
        return []
    
    batch_jobs = []
    
    for main_path in main_files:
        filename = os.path.basename(main_path)
        
        parts = filename.split('_')
        ticker = parts[0]
        date = parts[1]
        
        # Auto-guess the MO path based on the MAIN path
        mo_path = os.path.abspath(f"../data/raw/{ticker}_NASDAQ/MO/{ticker}_{date}.mat")
        
        if os.path.exists(mo_path):
            batch_jobs.append((main_path, mo_path))
        else:
            print(f"WARNING: Could not find MO file for {filename}. Skipping this day.")
            
    return batch_jobs