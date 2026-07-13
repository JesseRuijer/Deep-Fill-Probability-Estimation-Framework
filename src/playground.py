#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:50:49 2026

@author: jesseruijer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

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

import pandas as pd


df = pd.DataFrame({
    'OrderID': [997, 998, 999],
    'InitialVolume': [100, 500, 1000]
})


df2 = pd.DataFrame({
    'OrderID2': [997, 998, 999],
    'InitialVolume2': [100, 500, 1000]
})

print("--- Original DataFrame ---")
print(df)
print(df.T)

test_dic = {
    'SpeedDeltaMidprice': [0.5, 1.2, 0.0],
    'SpeedLogVolAhead': [10.5, 8.1, 4.2],
    'JustSomeBacon': [7, 7, 7]
}

# df = df.assign(**test_dic)

# print("\n--- After Unpacking Dictionary ---")
# print(df)

pd.concat([df, df2])


b = np.array([1,2,3,4,5])
print(min(b))
print(np.min(b))
print(np.minimum(b))







sweep_tods = [1,2,3]

final_tods = [2,3,4]









# 2. Find the most recent sweep for every single event/heartbeat
sweep_indices = np.searchsorted(sweep_tods, final_tods, side='right') -1
print(sweep_indices)

plt.plot([0,1], [1,0])


print(config.LGBM_MODEL_FEATURES)

a = np.arange(0,0.401, 0.01)
b = np.arange(0.43,1, 0.03)

print(a)
print(b)
print(np.concatenate([a,b]))




print(np.arange(1,3,1 ))

lst = [1,2,3,4]
print(lst[:-1])
print(lst[-1])


#######Start building a basic FNN on MNIST data to recognize digits from bad handwritten ones in pixel format
#### just to familiearize myself with pytorch and the concepts etc

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

##Device configuration
#Training is much faster on gpu, but apple silicon has MPS and if its on a different deivce you have to use NVIDIA CUDA if available (like on the cluster probs)
#this if else statement make sure to try and train on gpu no matter the device
#CPU is like a smart professor but just one guy and GPU is like 10000 dumb students, for neural net training whihc is just LAG, the dumb students are much faster

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Training on Apple Silicon MPS')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('Training on NVIDIA GPU (CUDA)')
else:
    device = torch.device('cpu')
print('Training on CPU')    

# hyperparameters

input_size = 784 # 28x28 pixels for each image, size of input layer
hidden_size = 100 # size of the single hidden layer
num_classes = 10 # cuz we have numbers from 0-9 so these are the amount of neurons in output layer, where each neuron represents the probability of the input number being that digit
epochs = 2 # number of full training stages on the entire dataset
batch_size = 100 # size on which it trains a bit then checks error then proceeds, faster than training on the full data set and then calculating error
learning_rate = 0.001 # weightsnew = weightsold - (learningrate * gradient)

#Importing MNIST data

train_dataset = torchvision.dataset.MNIST(root = './mnistdata', train = True, transform = transforms.ToTensor(), download = True)

test_dataset = torchvision.dataset.MNIST(root = './mnistdata', train = False, transform = transforms.ToTensor(), download = False)

train_loader = torch.utils.data.Dataloader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.Dataloader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

#Look at example of what gets pushed to the Net
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)





