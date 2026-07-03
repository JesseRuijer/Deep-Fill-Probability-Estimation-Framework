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

#For Shap and importance plots you have to use the base model 

def plot_lgbm_importances(base_model, features):
    #Visualising the important features in LGBM
    print("\n--- Extracting LightGBM Feature Importances ---")
    
    # LightGBM stores how many times a feature was used to split the data
    importances = base_model.booster_.feature_importance(importance_type = 'gain')
    
    #Double check the basic model initialized with the importance type gain 
        
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance (Gain)': importances
    }).sort_values(by='Importance (Gain)', ascending=False)
    
    # Plotting the results
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance (Gain)', y='Feature', data=importance_df, palette='viridis')
    plt.title('LightGBM Feature Importances (Total Information Gain)')
    plt.xlabel('Total Gain (Reduction in LogLoss)')
    plt.tight_layout()
    plt.show()
    
    return importance_df
    

def test_model(test_data, base_model, calibrated_model, scalar, model_name, features, is_multi):
    
    print(f'Evaluating {model_name}')
    
        
    X_test = test_data[features]
    y_raw = test_data[config.TARGET]
    weights = test_data['UnitWeight']
    
    
    #Check if the model needs scaling
    if scalar is not None:
        X_test_final = scalar.transform(X_test) #Here we transform and not fit anymore i.e we use same scale as above so comparisons are valid
    else:
        X_test_final = X_test
        
        
    if model_name == 'Logistic Regression':
        active_model = calibrated_model
        y_pred_prob_vis = calibrated_model.predict_proba(X_test_final)[:, 1]
                                
    elif model_name == 'Light Gradient Boosted Model':
        active_model = calibrated_model
        y_pred_prob_vis = calibrated_model.predict_proba(X_test_final)[:, 1]
    
    #Predict Probabilities
    
    if is_multi:
        #Since primary objective is Fill it squashes the expired and canceled down both being 0 and fill being 1
        y_pred_prob = active_model.predict_proba(X_test_final)[:, 1]
        y_true = np.where(y_raw == 1, 1, 0)
    
    else:
        y_pred_prob = active_model.predict_proba(X_test_final)[:, 1]
        y_true = y_raw
    
    #Evaluate performance metrics 
    
    dummy_fill_prob = np.average(y_true, weights = weights)
    
    print(f"{model_name} Engine metrics")
     
    brierscore = brier_score_loss(y_true, y_pred_prob, sample_weight = weights)
    print(f'Brier score is {brierscore:.3f}')
    
    brierskillscore = 1 - ((brierscore)/(dummy_fill_prob*(1-dummy_fill_prob)))
    print(f'Brier Skill Score is {brierskillscore}')
    
     
    logloss = log_loss(y_true, y_pred_prob, sample_weight = weights)
    print(f'Logloss score is {logloss:.3f}')
     
    #However i think AUC is only reliable on balanced data which this totally isnt, so the AUC is artificially inflated, why because we rarely ever have a fill and AUC is area under ROC, ROC formula is 
     
    #aucscore_lgbm = roc_auc_score(y_true, y_pred_prob_lgbm)
    #print(f'AUC score is {aucscore_lgbm:.3f}')
     
    avgprecision = average_precision_score(y_true, y_pred_prob, sample_weight = weights)
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
   
    
    dummy_y_pred_prob = np.full(len(y_true), dummy_fill_prob) #just creates an array of length y true with dummy fill probs
    print(f"Baseline Fill percentage is {(dummy_fill_prob * 100):.4f}% \n where the Dummy Fill prob is a measure of the total volume that got placed throughout the day that eventually resulted in a fill, rather than a simple counter of the individual order tickets%")
    
    
    #because were making a probability engine using logistic regression we must look at brier score and log loss to evaulte it
    #Confusion matrices and precisions for ex dont make too much sense here since then you need to define a treshold for when a probability gets put in the category
    #0 or 1 where for us that doesnt matter we're just interested in the pure probability of an order beig filled
    
    #Visualisiton of performance and comparison to baseline dummy model which just guesses a baseline percentage on each order for it being filled 
    #Dummy y fill prob is just an array of length y true with all entries equal to dummy fill prob
    
    dummy_brierscore = brier_score_loss(y_true, dummy_y_pred_prob, sample_weight = weights)
    print(f'Dummy Brier score is {dummy_brierscore:.3f}')
    
    dummy_logloss = log_loss(y_true, dummy_y_pred_prob, sample_weight = weights)
    
    print(f' Dummy Logloss score is {dummy_logloss:.3f}')
    
    avgprecision_dummy = average_precision_score(y_true, dummy_y_pred_prob, sample_weight = weights)
    print(f'Avg precision score is {avgprecision_dummy:.3f}')
    
    print(f'{model_name} performed {avgprecision/avgprecision_dummy:.3f} times better on PR AUC than dummy ')
    
    #Visualisation of performance vs dummy
    fig, axes = plt.subplots(2,2, figsize = (24,14))
    #calibration curve
    
    
    
    #manually writing a weighted calibration curve since weights isnt a supported parameter in sklearn calibration_curve
    
    def weighted_calibration_curve(y_true, y_pred, weights, n_bins=10):
        
        #So for this curve it groups into 10 bins the predict probability, but thats regardless of whether the option was alive for 1ms or for 10000ms so it doesnt say too much yet
        
        """
        Manually calculates a volume-weighted calibration curve using quantiles.
        """
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'weight': weights})
        
        # Sort by prediction to create quantile bins
        df = df.sort_values('y_pred')
        
        # Create quantile bins (duplicates='drop' prevents errors if many preds are exactly 0)
        df['bin'] = pd.qcut(df['y_pred'], q=n_bins, duplicates='drop')  #quantile cut cuts the data into equal amount of rows
        
        # Calculate the weighted average for both the actuals and the predictions per bin
        grouped = df.groupby('bin', observed=False) #observed is false is just to prevent pds error
        
        weighted_actual = grouped.apply(lambda x: np.average(x['y_true'], weights=x['weight'])) #apply is just pandas for applying a specific function, here we use once so use lambda, where we calculate weighted average of y_true and y_pred
        weighted_predicted = grouped.apply(lambda x: np.average(x['y_pred'], weights=x['weight']))
        
        # Drop NaNs just in case a bin was completely empty
        return weighted_actual.dropna().values, weighted_predicted.dropna().values
    
    engine_true, engine_prob_pred = weighted_calibration_curve(y_true, y_pred_prob_vis, weights, n_bins=10) #tuple unpacking since the function returns two variables, we just name them immediately in one line
    axes[0,0].plot([0,1], [0,1], color = 'grey', label = "Perfect Calibration") # axes[0] means we're talking about the left figure then [0,1] , [0,1] are x list and y list and are read vertically so the first point is 0,0 and the second point is 1,1 and a line is drawn between them i.e the perfect prediction line i think but check this
    axes[0,0].plot(engine_prob_pred, engine_true, color = 'b' ,label = f'{model_name}')
    axes[0,0].set_title('Calibration curve')
    axes[0,0].set_xlim(0, 0.5)
    axes[0,0].set_ylim(0, 0.5)
    axes[0,0].set_xlabel('Average Predicted Probability of Fill')
    axes[0,0].set_ylabel('Average Actual Fill Rate')
    axes[0,0].legend()
    
    
    def temporal_prob_curve(y_true, y_pred, weights, n_bins = 10):
        
        time_since_placement = test_data['TimeSincePlacement'].values
        
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'weights': weights,
            'time': time_since_placement})
        
        df = df.sort_values('time')
        
        df['time_bin'] = pd.qcut(df['time'], q = n_bins, duplicates = 'drop')
        
        grouped = df.groupby('time_bin', observed = False) #observed is false means even if there were zero orders in a bin, still use it
        
        avg_time = grouped['time'].mean().values
    
        weighted_actual = grouped.apply(lambda x: np.average(x['y_true'], weights = x['weights'])).values
        weighted_pred = grouped.apply(lambda x: np.average(x['y_pred'], weights = x['weights'])).values
  
        return avg_time, weighted_actual, weighted_pred  
  
    avg_time, temp_acc, temp_pred = temporal_prob_curve(y_true, y_pred_prob_vis, weights)
    
    axes[0,1].plot(avg_time, temp_acc, color = 'b', label = 'Actual fill rate')
    axes[0,1].plot(avg_time, temp_pred, color = 'r', label = f'{model_name} prediction')
    axes[0,1].set_title('Fill prob during lifetime of order')
    axes[0,1].set_xlabel('Time since placement') 
    axes[0,1].set_ylabel('P(fill)')
    axes[0,1].legend()
  
    
    # #Roc curve
    
    # engine_fpr, engine_tpr, tresholds = roc_curve(y_true, y_pred_prob_vis) #returns false postive rates and true positive rates, treshold which i think is the number or prob above or below it gives a certain classification
    # dummy_fpr, dummy_tpr, tresholds = roc_curve(y_true, dummy_y_pred_prob)
    # axes[0,1].plot(engine_fpr, engine_tpr, color = 'b', label = f'{model_name}')
    # axes[0,1].plot(dummy_fpr, dummy_tpr, color = 'r', label = 'Dummy')
    # axes[0,1].set_title('ROC curve (not useful for this type of data)')
    # axes[0,1].set_xlabel('Cancels or Expirations identified Falsely as Fills / All Actual Cancels') # False Postive rate = 1 - specificity = 1 - 
    # axes[0,1].set_ylabel('Fills correctly identified as Fills / All Actual Fills')  #Recall = TP /(TP + FN) = Sensitivity = True Positive Rate
    # axes[0,1].legend()
    
    #Precision recall curve 
    # a point of x = 0.16, y = 0.4 means the following: if we set our treshold to 16% so out of all the true fills
    # we only capture 16% then out of all the  orders our model flagged as a fill, 40% were actual fills , so when we captured only 16 % our treshold for selecting a fill was high, but you cant see that directly from the graph i think, look into a way of doing that 
    
    engine_precision, engine_recall, engine_treshold = precision_recall_curve(y_true, y_pred_prob_vis, sample_weight = weights)
    axes[1,0].plot(engine_recall, engine_precision, color = 'b', label = f'{model_name} PR')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].plot([0,1], [dummy_fill_prob, dummy_fill_prob], color = 'red', label = 'Dummy') #the dummy PR is just the baseline fill rate i.e here just a horizontal line
    axes[1,0].set_title('Precision-Recall curve')
    axes[1,0].set_ylabel('Fills Correctly Identified / All Orders Predicted To Fill')   #Precision = TP/(TP + FP)
    axes[1,0].set_xlabel('Fills correctly identified as Fills / All Actual Fills')  #Recall = TP /(TP + FN) = Sensitivity = True Positive Rate
    axes[1,0].legend()
        
    
    #Plot predicted probability distriubiton, use log scale on y axis since many more cancels then fills
    
    plot_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred_prob,
        'weights': weights
        })
    
    cancels_df = plot_df[plot_df['y_true'] == 0]
    fills_df = plot_df[plot_df['y_true'] == 1]
    
    axes[1, 1].hist(cancels_df['y_pred'], bins=50, alpha=0.3, color='red', label='Actual Cancels/Expires (0)', log = True, weights = cancels_df['weights'])
    axes[1, 1].hist(fills_df['y_pred'], bins=50, alpha=0.3, color='green', label='Actual Fills (1)', log = True, weights = fills_df['weights'])
    axes[1, 1].set_title(f'Distribution of Predicted Probabilities (Log Scale) for {model_name}')
    axes[1, 1].set_xlabel('Predicted Probability of Fill')
    axes[1, 1].set_ylabel('Vol of Orders (Log Scale)')
    axes[1, 1].legend()
    
    plt.show()
    
    #Plots Correlation heat plot of all features and of features that appear in model
    
    corr_matrix = X_test[features].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"All Feature Correlation Matrix of {model_name}")
    plt.show()
    



    def eval_at_heartbeat(test_data, y_true, y_pred_prob, weights, dummy_fill_prob):

        eval_df = pd.DataFrame({
            'y_true': y_true.values,
            'y_pred': y_pred_prob,
            'weight': weights.values,
            'ID':     test_data['ID'].values,
            'tsp':    test_data['TimeSincePlacement'].values,
        })
    
        # How many full heartbeat intervals have elapsed since placement
        eval_df['hb_idx'] = (eval_df['tsp'] // config.HEARTBEAT_INTERVAL).astype(int).clip(lower=0) #clip just sets a level where lower here sets the floor of the numbers at 0

        eval_df['EventualOutcome'] = eval_df['y_true']
    
        # Normalize each order's time axis to [0, 1]
        eval_df['NormTime'] = eval_df.groupby('ID')['tsp'].transform(
            lambda x: x / x.max() if x.max() > 0 else 0.0
        )
        eval_df['TimeBucket'] = pd.cut(eval_df['NormTime'], bins=20, labels = np.linspace(0, 1, 20))
    
        # Metric 1: Brier / BSS / LogLoss per heartbeat index
        naive_bs = dummy_fill_prob * (1 - dummy_fill_prob)
        results = {}
        for t, grp in eval_df.groupby('hb_idx'):
            y, p, w = grp['y_true'], grp['y_pred'], grp['weight']
            
            if y.nunique() < 2: #There might be some heartbeats where there are no fills and logloss needs both classes to perform so if thats the case for this specific heartbeat then we skip the logloss calculation
                continue
            
            bs = brier_score_loss(y, p, sample_weight=w)
            results[t] = {
                'brier':       bs,
                'brier_skill': 1 - bs / naive_bs,  
                'logloss':     log_loss(y, p, sample_weight=w),
            }

        heartbeat_metrics = pd.DataFrame(results).T
    
        # Metric 2: avg predicted prob over lifetime, split by outcome
        
        trajectory = (
            eval_df.groupby(['TimeBucket', 'EventualOutcome']).apply(lambda x: np.average(x['y_pred'], weights = x['weight'])).unstack()    #unstack immediately gives back the outcome column in two seperate columns
            )
    
        # Metric 3: Brier score over normalized lifetime 
        brier_by_time = eval_df.groupby('TimeBucket').apply(
            lambda g: brier_score_loss(g['y_true'], g['y_pred'], sample_weight=g['weight'])
        )
    
        # Plots
    
        # BSS per heartbeat 
        plt.figure(figsize = (20,10))
        plt.plot(heartbeat_metrics.index, heartbeat_metrics['brier_skill'], color='b')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1,
                        label='Dummy baseline (BSS = 0)')
        plt.xlabel('Heartbeat index (intervals since placement)')
        plt.ylabel('Brier Skill Score')
        plt.title(f'Does more LOB info improve accuracy? {model_name}')
        plt.legend()
        plt.show()
        
        
        #For this plot the orders labelled as filled were just any order that has a (partial) fill, there was no distinction between them, but for the 
        plt.figure(figsize = (20,10))
        # Trajectory by eventual outcome
        if 1 in trajectory.columns:
            plt.plot(trajectory[1], 'o-',  color='g',  label='Eventually filled')
        if 0 in trajectory.columns:
            plt.plot(trajectory[0], 'o--', color='r', label='Eventually canceled')
        plt.xlabel('Normalized order lifetime (0=placement, 1=death)')
        plt.ylabel('Avg p(fill)')
        plt.title(f'Predicted prob over lifetime by eventual outcome {model_name}')
        plt.legend()
    
        # Brier over normalized lifetime with naive baseline for reference
        plt.figure(figsize = (20,10))
        plt.plot(brier_by_time, 'o-', color='#D85A30', label='Model')
        plt.axhline(naive_bs, color='gray', linestyle='--',
                        linewidth=1, label=f'Naive BS ({naive_bs:.3f})')
        plt.xlabel('Normalized order lifetime')
        plt.ylabel('Brier score')
        plt.title(f'Brier score over order lifetime {model_name}')
        plt.legend()

        plt.show()
        
        #The heartbeat metrics list gives the three metrics for all orders that survived the row number amount of heartbeats
        #For example the first row is the performance of the model across every single order the moment it was placed
        #The 10th row is the performance of the model evaluated after 10 heartbeats on the orders that still remain in the LOB after 10 heartbeats 
    
        print(heartbeat_metrics.to_string())
        return heartbeat_metrics

    eval_at_heartbeat(test_data, y_true, y_pred_prob, weights, dummy_fill_prob)

    

#Making performance plots
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred_prob,
        'weights': weights,
        'dummy_pred': dummy_y_pred_prob
        })
    df = df.sort_values('y_pred')
    
    deltap = 0.01
    
    bins_low = np.arange(0,0.401, deltap)
    bins_high = np.arange(0.43,1, 3*deltap)
    
    bins_custom = np.concatenate((bins_low, bins_high))
    df['bins'] = pd.cut(df['y_pred'], bins = bins_custom)
    
    grouped = df.groupby('bins', observed = False)
    
    def safe_weigths(x, y):
        if x['weights'].sum() == 0:
            return np.nan
        return np.average(x[y], weights = x['weights'])
    
    weighted_actual = grouped.apply(lambda x: safe_weigths(x, 'y_true')).dropna()
    weighted_pred = grouped.apply(lambda x: safe_weigths(x, 'y_pred')).dropna() 
    weighted_dummy = np.average(y_true, weights = weights)

    print(f'weighetd dummy {weighted_dummy}')
    
    plt.plot(weighted_pred, weighted_actual, color = 'b', label = 'Model')
    plt.plot(weighted_dummy,weighted_dummy, marker = 'o', markersize = 10, markeredgecolor = 'black', color = 'yellow', label = f'dummy {weighted_dummy:.2f}%')
    plt.plot([0,1], [0,1], color = 'black', label = 'Perfect')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Predicted Fill Prob')
    plt.ylabel('Actual Fill Prob')
    plt.title(f'Performance of {model_name}')
    plt.legend()
    plt.show()

    #Similar plot to above but now for orders only placed at best bid and best ask
    
    mask = test_data['DistanceToTouch'] == 0
    
    y_true_filter = y_true[mask]
    y_pred_prob_filter = y_pred_prob[mask]
    weights_filter = weights[mask]
    dummy_y_pred_prob_filter = np.average(y_true_filter, weights = weights_filter)
    
    df_filter = pd.DataFrame({
        'y_true':  y_true_filter,
        'y_pred': y_pred_prob_filter,
        'weights':  weights_filter,
        'dummy_pred': dummy_y_pred_prob_filter 
        })
    
    df_filter['bins'] = pd.cut(df_filter['y_pred'], bins = bins_custom)
    
    grouped = df_filter.groupby('bins', observed = False)
    
    weighted_actual = grouped.apply(lambda x: safe_weigths(x, 'y_true')).dropna()
    weighted_pred = grouped.apply(lambda x: safe_weigths(x, 'y_pred')).dropna() 
    weighted_dummy = np.average(y_true_filter, weights = weights_filter)
    
    print(f'weighetd dummy mask {weighted_dummy}')
    
    plt.plot(weighted_pred, weighted_actual, color = 'b', label = 'Model')
    plt.plot(weighted_dummy, weighted_dummy, marker = 'o', markersize = 10, markeredgecolor = 'black', color = 'yellow', label = f'dummy {weighted_dummy:.2f}%')
    plt.plot([0,1], [0,1], color = 'black', label = 'Perfect')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Predicted Fill Prob')
    plt.ylabel('Actual Fill Prob')
    plt.title(f'Performance of {model_name} On Orders only placed at Best Bid or Best Ask')
    plt.legend()
    plt.show()
    
        
    #Make a linegraph with on x axis the bins and on y axis the vol of orders landed in that bin 
    
    plt.figure(figsize = (20,10))
    vol_per_bin = df.groupby('bins', observed = False)['weights'].sum()
    
    middle = [b.mid for b in vol_per_bin.index]
    
    plt.plot(middle, vol_per_bin, color = 'b', label = 'LO Vol')
    plt.xlabel('Probability Bins')
    plt.xlim(0, 1)
    plt.ylabel('Vol of LOs')
    plt.title(f'Amount Of LO Vol appearing in each predictive probability bin {model_name}')
    plt.legend()
    plt.show()
    
    
    
        
    if model_name == "Light Gradient Boosted Model":
        plot_lgbm_importances(base_model, features)









        

# def predict_order_fill_prob(features):
#     #predicts specific probability for a given limit order being filled using the logistic regression engine from above
#     #since pd dfs are slow its better to use np array here
    
#     input_array = np.array(features).reshape(1,-1) #resshape needed 1 means passing 1 row, -1 means calculate the right dim for the columns so this creates a matrix which is whats needed for sci kit later
#     scaled_input  = (input_array - scalar.mean_) / scalar.scale_ # I think the trailing _ tells sci kit to look at the fitted values and calculate mean and std of those                   using scalar so we do the standardizations for each value and not all at the same time
#     fill_prob = calibrated_model.predict_proba(scaled_input)[0,1]
    
#     return fill_prob
#     #Better to also create a function that extracts these features for maybe a given order ID?
    
    
    
    
    
    
    
    
    