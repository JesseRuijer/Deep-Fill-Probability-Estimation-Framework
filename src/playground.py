#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:50:49 2026

@author: jesseruijer
"""

import pandas as pd
import numpy as np

test_df = pd.DataFrame({
    
    'a': [1,2,3],
    'b': [4,5,6]
    
    })

print(test_df.shift(1))


test_arr1 = np.array([1,2,3])
print(test_arr1)

test_arr2 = np.array([4,0,5])
print(test_arr2)


speed_array = np.divide(
            test_arr1, 
            test_arr2, 
            out=np.zeros_like(test_arr1, dtype = float), 
            where=(test_arr2 != 0)
        )

print(speed_array)

