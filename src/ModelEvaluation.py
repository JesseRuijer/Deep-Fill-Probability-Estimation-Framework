#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:17:44 2026

@author: jesseruijer
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss, 
    log_loss, 
    roc_auc_score, 
    average_precision_score, 
    roc_curve, 
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

import config 


def plot_lgbm_importances(base_model, features):
    #Visualising the important features in LGBM
    print("\n--- Extracting LightGBM Feature Importances ---")
    
    # LightGBM stores how many times a feature was used to split the data
    importances = base_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance (Splits)': importances
    }).sort_values(by='Importance (Splits)', ascending=False)
    
    # Plotting the results
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance (Splits)', y='Feature', data=importance_df, palette='viridis')
    plt.title('LightGBM Feature Importances (Tree Splits)')
    plt.xlabel('Number of times feature was used to split data')
    plt.tight_layout()
    plt.show()
    
    return importance_df

def test_model(test_data, base_model, calibrated_model, scalar, model_name, features, is_multi):
    
    print(f'Evaluating {model_name}')
    
        
    X_test = test_data[features]
    y_raw = test_data[config.TARGET]
    
    #Check if the model needs scaling
    if scalar is not None:
        X_test_final = scalar.transform(X_test) #Here we transform and not fit anymore i.e we use same scale as above so comparisons are valid
    else:
        X_test_final = X_test
    
    #Predict Probabilities
    
    if is_multi:
        #Since primary objective is Fill it squashes the expired and canceled down both being 0 and fill being 1
        y_pred_prob = calibrated_model.predict_proba(X_test_final)[:, 1]
        y_true = np.where(y_raw == 1, 1, 0)
    
    else:
        y_pred_prob = calibrated_model.predict_proba(X_test_final)[:, 1]
        y_true = y_raw
    
    #Evaluate performance metrics 
    print(f"{model_name} Engine metrics")
     
    brierscore = brier_score_loss(y_true, y_pred_prob)
    print(f'Brier score is {brierscore:.3f}')
     
    logloss = log_loss(y_true, y_pred_prob)
    print(f'Logloss score is {logloss:.3f}')
     
    #However i think AUC is only reliable on balanced data which this totally isnt, so the AUC is artificially inflated, why because we rarely ever have a fill and AUC is area under ROC, ROC formula is 
     
    #aucscore_lgbm = roc_auc_score(y_true, y_pred_prob_lgbm)
    #print(f'AUC score is {aucscore_lgbm:.3f}')
     
    avgprecision = average_precision_score(y_true, y_pred_prob)
    print(f'Avg precision score is {avgprecision:.3f}')
    
    #Odds ratios if model is logistic regression
    
    if model_name == "Logistic Regression":
        model_coef_df = pd.DataFrame(
                {
                 "Feature": features,
                 "Coefficient (Log Odds)" : base_model.coef_[0],  # We only predict binary classification so our model only has one row so access that with [0]
                 "Odds Ratio": np.exp(base_model.coef_[0])
                 }
                )
        print(f' Logistic Regression ORs \n {model_coef_df.sort_values(by = "Odds Ratio", key=abs, ascending=False)}')
    
    
    ################## Dummy Model #################################
    
    print("Dummy metrics")
    
    #Baseline fill percentage which i defined as the number of ones divided by number of ones and zeros in fill_map, which guarantees uniqueness by the fact i used .last in code before it
   
    dummy_fill_prob = np.average(y_true, weights = test_data['UnitWeight'])
    dummy_y_pred_prob = np.full(len(y_true), dummy_fill_prob) #just creates an array of length y true with dummy fill probs
    print(f"Baseline Fill percentage is {(dummy_fill_prob * 100):.4f} %")
    
    
    #because were making a probability engine using logistic regression we must look at brier score and log loss to evaulte it
    #Confusion matrices and precisions for ex dont make too much sense here since then you need to define a treshold for when a probability gets put in the category
    #0 or 1 where for us that doesnt matter we're just interested in the pure probability of an order beig filled
    
    #Visualisiton of performance and comparison to baseline dummy model which just guesses a baseline percentage on each order for it being filled 
    #Dummy y fill prob is just an array of length y true with all entries equal to dummy fill prob
    
    dummy_brierscore = brier_score_loss(y_true, dummy_y_pred_prob)
    print(f'Dummy Brier score is {dummy_brierscore:.3f}')
    
    dummy_logloss = log_loss(y_true, dummy_y_pred_prob)
    print(f' Dummy Logloss score is {dummy_logloss:.3f}')
    
    avgprecision_dummy = average_precision_score(y_true, dummy_y_pred_prob)
    print(f'Avg precision score is {avgprecision_dummy:.3f}')
    
    #Visualisation of performance vs dummy
    fig, axes = plt.subplots(2,2, figsize = (24,14))
    #calibration curve
    engine_true, engine_prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10, strategy = 'quantile') #tuple unpacking since the function returns two variables, we just name them immediately in one line
    axes[0,0].plot([0,1], [0,1], color = 'grey', label = "Perfect Calibration") # axes[0] means we're talking about the left figure then [0,1] , [0,1] are x list and y list and are read vertically so the first point is 0,0 and the second point is 1,1 and a line is drawn between them i.e the perfect prediction line i think but check this
    axes[0,0].plot(engine_prob_pred, engine_true, color = 'b' ,label = f'{model_name}')
    axes[0,0].set_title('Calibration curve')
    axes[0,0].set_xlabel('Average Predicted Probability of Fill')
    axes[0,0].set_ylabel('Average Actual Fill Rate')
    axes[0,0].legend()
    
    #Roc curve
    
    engine_fpr, engine_tpr, tresholds = roc_curve(y_true, y_pred_prob) #returns false postive rates and true positive rates, treshold which i think is the number or prob above or below it gives a certain classification
    dummy_fpr, dummy_tpr, tresholds = roc_curve(y_true, dummy_y_pred_prob)
    axes[0,1].plot(engine_fpr, engine_tpr, color = 'b', label = f'{model_name}')
    axes[0,1].plot(dummy_fpr, dummy_tpr, color = 'r', label = 'Dummy')
    axes[0,1].set_title('ROC curve (not useful for this type of data)')
    axes[0,1].set_xlabel('Cancels or Expirations identified Falsely as Fills / All Actual Cancels') # False Postive rate = 1 - specificity = 1 - 
    axes[0,1].set_ylabel('Fills correctly identified as Fills / All Actual Fills')  #Recall = TP /(TP + FN) = Sensitivity = True Positive Rate
    axes[0,1].legend()
    
    #Precision recall curve
    engine_precision, engine_recall, engine_treshold = precision_recall_curve(y_true, y_pred_prob)
    axes[1,0].plot(engine_recall, engine_precision, color = 'b', label = f'{model_name} PR')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].plot([0,1], [dummy_fill_prob, dummy_fill_prob], color = 'red', label = 'Dummy') #the dummy PR is just the baseline fill rate i.e here just a horizontal line
    axes[1,0].set_title('Precision-Recall curve')
    axes[1,0].set_ylabel('Fills Correctly Identified / All Orders Predicted To Fill')   #Precision = TP/(TP + FP)
    axes[1,0].set_xlabel('Fills correctly identified as Fills / All Actual Fills')  #Recall = TP /(TP + FN) = Sensitivity = True Positive Rate
    axes[1,0].legend()
        
    
    #Plot predicted probability distriubiton, use log scale on y axis since many more cancels then fills
    
    axes[1, 1].hist(y_pred_prob[y_true == 0], bins=50, alpha=0.3, color='red', label='Actual Cancels/Expires (0)', log=True)
    axes[1, 1].hist(y_pred_prob[y_true == 1], bins=50, alpha=0.3, color='green', label='Actual Fills (1)', log=True)
    axes[1, 1].set_title(f'Distribution of Predicted Probabilities (Log Scale) for {model_name}')
    axes[1, 1].set_xlabel('Predicted Probability of Fill')
    axes[1, 1].set_ylabel('Number of Orders (Log Scale)')
    axes[1, 1].legend()
    
    plt.show()
    
    #Plots Correlation heat plot of all features and of features that appear in model
    
    corr_matrix = X_test[features].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"All Feature Correlation Matrix of {model_name}")
    plt.show()
        
    if model_name == "Light Gradient Boosted Model Regression":
        plot_lgbm_importances(base_model, features)
        

# def predict_order_fill_prob(features):
#     #predicts specific probability for a given limit order being filled using the logistic regression engine from above
#     #since pd dfs are slow its better to use np array here
    
#     input_array = np.array(features).reshape(1,-1) #resshape needed 1 means passing 1 row, -1 means calculate the right dim for the columns so this creates a matrix which is whats needed for sci kit later
#     scaled_input  = (input_array - scalar.mean_) / scalar.scale_ # I think the trailing _ tells sci kit to look at the fitted values and calculate mean and std of those                   using scalar so we do the standardizations for each value and not all at the same time
#     fill_prob = calibrated_model.predict_proba(scaled_input)[0,1]
    
#     return fill_prob
#     #Better to also create a function that extracts these features for maybe a given order ID?
    
    
    
    
    
    
    
    
    