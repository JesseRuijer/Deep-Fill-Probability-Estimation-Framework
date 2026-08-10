#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:05:38 2026

@author: jesseruijer
"""
"""

This script just handles the data entries into the programme via finder instead of having to manually do this (avoids typos and reduces time)
Also codes a cache_file so if wanting to rerun the same data multiple times the finder doesnt keep on opening
Note majority of this code was AI generated and changed by me to fit as I have no expertise in software engineering related to File managing on pc etc

"""

import tkinter as tk
from tkinter import filedialog
import os
import json
import sys
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_FILE = SCRIPT_DIR / ".last_paths.json"

def select_files_via_finder() -> tuple[str | None, str | None]:
    
    """
    GUI for selecting the files via finder
    """
    
    root = tk.Tk()
    root.withdraw() 
    
    print("Opening Finder for selecting Main data set")
    main_file = filedialog.askopenfilename(title="Select the MAIN .mat file", filetypes=[("MATLAB files", "*.mat")])
    if not main_file: return None, None
        
    print("Opening Finder for selecting MO data set")
    mo_file = filedialog.askopenfilename(title="Select the MO .mat file", filetypes=[("MATLAB files", "*.mat")])
    
    return main_file, mo_file

def get_data_paths() -> tuple[str, str]:
    
    """
    If you want to use new data or remain with the old data, i.e retrieves paths from cache or user can put in new paths via Finder
    """
    
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            saved_paths = json.load(f)
            
        main_exists = Path(saved_paths['main_path']).exists()
        mo_exists = Path(saved_paths['mo_path']).exists()
        
        if main_exists and mo_exists:
            print(f"MAIN: {os.path.basename(saved_paths['main_path'])}")
            print(f"MO:   {os.path.basename(saved_paths['mo_path'])}")
            
            user_choice = input("Press ENTER to use these files, or type 'n' to pick a new one(s): ")
            
            if user_choice.strip().lower() != 'n':
                return saved_paths['main_path'], saved_paths['mo_path']
            
        else:
            print("\n[WARNING] Cached paths do not exist on this machine")

    main_path, mo_path = select_files_via_finder()
    
    if main_path and mo_path:
        with open(CACHE_FILE, 'w') as f:
            json.dump({'main_path': main_path, 'mo_path': mo_path}, f)
            
    return main_path, mo_path

def generate_dynamic_paths(main_file_path: str) -> Path:
    
    """
    Obtaining path from parquet files
    """
    
    filename = os.path.basename(main_file_path)
    parts = filename.split('_')
    
    ticker = parts[0]
    date = parts[1]
    formatted_date = f"{date[:4]}_{date[4:6]}_{date[6:]}"
    
    #pathlib way
    #anchor to script
    script_dir = Path(__file__).resolve().parent
    
    #build path, up one level then into data and into procesed
    processed_dir = script_dir.parent / 'data' / 'processed'
    
    #failsafe, create folders if they dont exist
    processed_dir.mkdir(parents = True, exist_ok = True)
    
    binary_out = processed_dir / f"{ticker}_BINARY_{formatted_date}.parquet"

    return binary_out

def get_ml_training_paths() -> dict[str, list[str]]:
    
    """
    Gives paths to the training and testing files
    """
    
    ML_CACHE_FILE = SCRIPT_DIR / ".ml_cache.json"

    if os.path.exists(ML_CACHE_FILE):
        with open(ML_CACHE_FILE, 'r') as f:
            saved_paths = json.load(f)
            
        cache_is_valid = True
        for key, value in saved_paths.items():
            if not value:
                continue
            paths_to_check = value if isinstance(value, list) else [value]
            if any(not Path(p).exists() for p in paths_to_check):
                cache_is_valid = False
                break
            
        if cache_is_valid:
            print("\n--- ML DATA CACHE FOUND ---")
            
            # .get() prevents errors if a slot is empty
            if saved_paths.get('train_bin'): 
                files = saved_paths['train_bin']
                for f in (files if isinstance(files, list) else [files]):
                    print(f"TRAIN (Bin):   {os.path.basename(f)}")
           
            if saved_paths.get('test_bin'): 
                files = saved_paths['test_bin']
                for f in (files if isinstance(files, list) else [files]):
                    print(f"TEST (Bin):   {os.path.basename(f)}")
            
            print("---------------------------")
            
            user_choice = input("Press [ENTER] to reuse these datasets, or type 'n' to pick new ones: ")
        
            if user_choice.strip().lower() != 'n':
                return saved_paths
        else:
            print("\n[WARNING] Cached paths do not exist on this machine")

    root = tk.Tk()
    root.withdraw()
    
    script_dir = Path(__file__).resolve().parent
    init_dir = script_dir.parent / 'data' / 'processed'
    
    #Tkinter needs raw strings for directories so convert it quickly
    
    init_dir_str = str(init_dir)
    
    
    print("\n--- SELECT DATA FOR MACHINE LEARNING ---")
    
    print("Highlight the TRAINING file(s) you want to use (Hold Cmd/Shift for multiple).")
    train_files = filedialog.askopenfilenames(initialdir=init_dir_str, title="Select TRAIN Data", filetypes=[("All files", "*.*")])
    
    if not train_files:
        return {} # Returns empty if canceled

    print("Highlight the TESTING file(s) you want to use (Hold Cmd/Shift for multiple).")
    test_files = filedialog.askopenfilenames(initialdir=init_dir, title="Select TEST Data", filetypes=[("All files", "*.*")])
    
    if not test_files:
        return {} # Returns empty if canceled

    # Set up empty slots
    paths = {
        'train_bin': [], 'test_bin': [], 
    }

    # Automatically sort the files you highlighted into the right slots
    for f in train_files:
        if "BINARY" in f.upper(): paths['train_bin'].append(f)

    for f in test_files:
        if "BINARY" in f.upper(): paths['test_bin'].append(f)
        
    #Sort the files in order, which works by our file labelling, so first entry in the list will be the first training day, second second trading day etc
    paths['train_bin'].sort()
    paths['test_bin'].sort()

    with open(ML_CACHE_FILE, 'w') as f:
        json.dump(paths, f)

    return paths

def get_batch_data_paths() -> list[tuple[str, str]]:
    
    """
    Gets strings of files to save from finder, also automatically finds the corresponding MO files to the event files 
    """
    
    batch_jobs = []
    script_dir = Path(__file__).resolve().parent
    raw_data_dir = script_dir.parent / 'data' / 'raw'
    
    #Cluster bypass so if you run main.py it bypasses the GUI (for executing on clusters that do not have GUI)
    
    if '--cluster' in sys.argv:
        print('Running in cluster mode')
        
        main_files = glob.glob(str(raw_data_dir / '*_NASDAQ/*.mat'))
        
        if not main_files:
            print('No .mat files found')
            return []
    
    else:  #local GUI mode
        root = tk.Tk()
        root.withdraw() 
        
        print("\n--- BATCH DATA INGESTION ---")
        print("Opening Finder... Highlight as many MAIN .mat files as you want.")
        
        main_files = filedialog.askopenfilenames(
            title="Select MAIN .mat files (Hold Cmd/Shift for multiple)", 
            filetypes=[("MATLAB files", "*.mat")]
        )
    
    if not main_files: 
        print('No .mat files found')
        return []
    
    batch_jobs = []
    
    for main_path in main_files:
        filename = os.path.basename(main_path)
        
        parts = filename.split('_')
        ticker = parts[0]
        date = parts[1]
        
        script_dir = Path(__file__).resolve().parent
        
        mo_path = script_dir.parent / 'data' / 'raw' / f'{ticker}_NASDAQ' / 'MO' / f'{ticker}_{date}.mat'  

        if mo_path.exists():
            batch_jobs.append((main_path, str(mo_path)))
        else:
            print(f"WARNING: Could not find MO file for {filename}. Skipping this day, was looking here: {mo_path}")
            
    return batch_jobs