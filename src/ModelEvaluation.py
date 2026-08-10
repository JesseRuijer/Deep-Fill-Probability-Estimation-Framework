#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:17:44 2026

@author: jesseruijer
"""

"""
Creates all model evaluation plots as well as calculation of evaluation metrics

"""

import pandas as pd
import torch.nn as nn
import numpy as np
import seaborn as sns
import shap
import torch 
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    brier_score_loss, 
    log_loss, 
    average_precision_score, 
    precision_recall_curve
)
import config 


# =============================================================================
# 1. LGBM & FNN EXPLAINABILITY (SHAP / GAIN)
# =============================================================================


def plot_lgbm_importances(base_model, features: list) -> pd.DataFrame:

    """
    Visualising the important features in LGBM
    For Shap and importance plots you have to use the base model 
    """
    
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

 
def plot_fnn_importances(model, features: list, test_data: pd.DataFrame, scalar) -> None:
    """
    Use SHAP to visualise FNN 'decisions'
    """
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    raw_data = test_data[features].values
    scaled_data = scalar.transform(raw_data)
    
    # Take a small, random sample for the background distribution
    # SHAP needs this to understand what "average" looks like
    background_idx = np.random.choice(len(scaled_data), 500, replace=False)
    X_background = torch.tensor(scaled_data[background_idx], dtype=torch.float32).to(device)
    
    # take a small, random sample of the data we actually want to explain
    test_idx = np.random.choice(len(scaled_data), 1000, replace=False)
    X_sample = torch.tensor(scaled_data[test_idx], dtype=torch.float32).to(device)
    
    # create a Custom Wrapper just for SHAP
    class PropWrapper(nn.Module):
        def __init__(self, base_model):
            super(PropWrapper, self).__init__()
            self.base_model = base_model
            
        def forward(self, x):
            
            logits = self.base_model(x)
           
            probs = torch.sigmoid(logits)
            return probs
            
    # Wrap the base PyTorch model
    raw_model = model.model.to(device)
    raw_model.eval()
    
    # Initialize Explainer with the background dataset
    explainer = shap.GradientExplainer(raw_model, X_background) 
    
    print("Calculating SHAP values")
    # Calculate SHAP values for the test sample
    shap_values = explainer.shap_values(X_sample)
    
    # DeepExplainer sometimes returns a list of arrays (one for each class). 
    # If it's a list, we grab the array for class 1 (Fills)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0] 
    # Also ensure our sample data is back on the CPU as a numpy array for the plot
    X_sample_np = X_sample.cpu().numpy()
    
    # Plot the results
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_sample_np, feature_names=features, show=False)
    plt.title('FNN SHAP Feature Importances (Limit Order Fill Probability)')
    plt.tight_layout()
    plt.show() 


def test_model(test_data: pd.DataFrame, base_model, calibrated_model, scalar, model_name: str, features: list, is_multi: bool) -> None:
    
    """Evaluate model performance via volume-weighted metrics and diagnostic plots.

    Generates the following plots:
        4-Panel Overview (fig, axes = plt.subplots(2, 2)):
            - Top-Left (Calibration Curve)
            - Top-Right (Temporal Probability): Tracks average fill probability as a function
              of clock time elapsed since initial placement (TimeSincePlacement).
            - Bottom-Left (Precision-Recall Curve)
            - Bottom-Right (Log-Scale Probability Distribution): Histograms predicted probability
              density separated by actual outcomes (Fills vs. Cancels/Expires).
         Heartbeat Diagnostics:
            - BSS per Heartbeat: Measures whether predictive skill degrades over time
              across 1,000ms heartbeat snapshots.
            - Alligator Plot (Normalized Lifetime): Traces predicted fill probability over
              an order's normalized life (0=placement, 1=death), separated by eventual fill/cancel.
            - Brier Score over Normalized Lifetime: Plots Brier score drift against naive baseline.
         Binned Probability Diagnostics:
            - Model vs. Actual Fill Probability across custom probability buckets for all orders
              and touch-only orders (DistanceToTouch == 0).
            - Volume per Bin: Log-scale distribution of LOB volume across prediction bins.
    """
    
    print(f'Evaluating {model_name}')
        
    X_test = test_data[features]
    y_raw = test_data[config.TARGET]
    weights = test_data['UnitWeight']
    
    #Check if the model needs scaling
    if scalar is not None:
        X_test_final = scalar.transform(X_test.values) #Here we transform and not fit anymore i.e we use same scale as above so comparisons are valid
    else:
        X_test_final = X_test.values
        
    active_model = calibrated_model if calibrated_model is not None else base_model
    y_pred_prob_vis = active_model.predict_proba(X_test_final)[:,1]
    
    
    #Predict Probabilities
    y_pred_prob = y_pred_prob_vis
    y_true = y_raw
    
    
# =============================================================================
# 2. Metrics Evaluation & Performance plots
# =============================================================================
    #Evaluate performance metrics 
    
    dummy_fill_prob = np.average(y_true, weights = weights)
    
    print(f"{model_name} Engine metrics")
     
    brierscore = brier_score_loss(y_true, y_pred_prob, sample_weight = weights)
    print(f'Brier score is {brierscore:.3f}')
    
    brierskillscore = 1 - ((brierscore)/(dummy_fill_prob*(1-dummy_fill_prob)))
    print(f'Brier Skill Score is {brierskillscore}')
    
     
    logloss = log_loss(y_true, y_pred_prob, sample_weight = weights)
    print(f'Logloss score is {logloss:.3f}')
    
    dummy_y_pred_prob = np.full(len(y_true), dummy_fill_prob) #just creates an array of length y true with dummy fill probs
    print(f"Baseline Fill percentage is {(dummy_fill_prob * 100):.4f}% \n where the Dummy Fill prob is a measure of the total volume that got placed throughout the day that eventually resulted in a fill, rather than a simple counter of the individual order tickets%")
    
    dummy_logloss = log_loss(y_true, dummy_y_pred_prob, sample_weight = weights)
    
    loglosskillscore = 1 - ((logloss)/(dummy_logloss))
    print(f'Logloss Skill Score is {loglosskillscore}')
     
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
   
    dummy_brierscore = brier_score_loss(y_true, dummy_y_pred_prob, sample_weight = weights)
    print(f'Dummy Brier score is {dummy_brierscore:.3f}')
    
    
    print(f' Dummy Logloss score is {dummy_logloss:.3f}')
    
    avgprecision_dummy = average_precision_score(y_true, dummy_y_pred_prob, sample_weight = weights)
    print(f'Avg precision score is {avgprecision_dummy:.3f}')
    
    print(f'{model_name} performed {avgprecision/avgprecision_dummy:.3f} times better on PR AUC than dummy ')
    
    #Visualisation of performance vs dummy
    fig, axes = plt.subplots(2,2, figsize = (24,14))
    
    
    def weighted_calibration_curve(y_true: pd.Series, y_pred: np.ndarray, weights: pd.Series, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
        
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
    axes[0,0].set_title('Calibration Curve')
    axes[0,0].set_xlim(0, 0.5)
    axes[0,0].set_ylim(0, 0.5)
    axes[0,0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0,0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0,0].grid(True, alpha = 0.3)
    axes[0,0].set_xlabel('Average Predicted Probability of Fill')
    axes[0,0].set_ylabel('Average Actual Fill Rate')
    axes[0,0].legend()
    
    
    def temporal_prob_curve(y_true: pd.Series, y_pred: np.ndarray, weights: pd.Series, n_bins:int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Function to 
        Create and plot fill probability true vs predicted over clock time
        """
        
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
    axes[0,1].set_title('Fill Probability During Lifetime')
    axes[0,1].set_xlabel('Time since placement (ms)') 
    axes[0,1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0,1].set_ylabel('P(fill)')
    axes[0,1].grid(True, alpha = 0.3)
    axes[0,1].legend()
    
    
    #precision-recall curve, weighted properly
    engine_precision, engine_recall, engine_treshold = precision_recall_curve(y_true, y_pred_prob_vis, sample_weight = weights)
    axes[1,0].plot(engine_recall, engine_precision, color = 'b', label = f'{model_name} PR')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].plot([0,1], [dummy_fill_prob, dummy_fill_prob], color = 'red', label = 'Dummy') #the dummy PR is just the baseline fill rate i.e here just a horizontal line
    axes[1,0].set_title('Precision-Recall curve')
    axes[1,0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1,0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1,0].grid(True, alpha = 0.3)
    axes[1,0].set_ylabel('Fills Correctly Identified / All Orders Predicted To Fill')   #Precision = TP/(TP + FP)
    axes[1,0].set_xlabel('Fills correctly identified as Fills / All Actual Fills')  #Recall = TP /(TP + FN) = Sensitivity = True Positive Rate
    axes[1,0].legend()
        
    
    #Plot histogram of predicted fill probs versus volume of orders
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
    axes[1,1].grid(True, alpha = 0.3)
    axes[1,1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].set_xlabel('Predicted Probability of Fill')
    axes[1, 1].set_ylabel('Vol of Orders (Log Scale)')
    axes[1, 1].legend()
    plt.show()
    
    #Plots Correlation heat plot of all features and of features that appear in model
    corr_matrix = X_test[features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"All Feature Correlation Matrix of {model_name}")
    plt.show()
    
# =============================================================================
# 3. Heartbeat performance plots
# =============================================================================

    def eval_at_heartbeat(test_data:pd.DataFrame, y_true: pd.Series, y_pred_prob: np.ndarray, weights: pd.Series, dummy_fill_prob: float) -> pd.DataFrame:
        
        """
        Creates performance plots to show how model performs depending on how many heartbeats were evaluated
        The heartbeat metrics list gives the three metrics for all orders that survived the row number amount of heartbeats
        For example the first row is the performance of the model across every single order the moment it was placed
        The 10th row is the performance of the model evaluated after 10 heartbeats on the orders that still remain in the LOB after 10 heartbeats 
        """

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
        plt.grid(True, alpha = 0.3)
        plt.ylabel('Brier Skill Score')
        plt.title(f'Does more LOB info improve accuracy? {model_name}')
        plt.legend()
        plt.show()
        
        
        #For this plot the orders labelled as filled were just any order that has a (partial) fill, there was no distinction between them, but they were correctly weighted 
        plt.figure(figsize = (20,10))
        # Trajectory by eventual outcome
        if 1 in trajectory.columns:
            plt.plot(trajectory[1], 'o-',  color='g',  label='Eventually filled')
        if 0 in trajectory.columns:
            plt.plot(trajectory[0], 'o-', color='r', label='Eventually canceled')
        plt.xlabel('Normalized order lifetime (0=placement, 1=death)')
        plt.ylabel('Average Fill Probability')
        plt.grid(True, alpha = 0.3)
        plt.title(f'Predicted Fill Probability over lifetime by eventual outcome {model_name}')
        plt.legend()
    
        # Brier over normalized lifetime with naive baseline for reference
        plt.figure(figsize = (20,10))
        plt.plot(brier_by_time, 'o-', color='#D85A30', label='Model')
        plt.axhline(naive_bs, color='gray', linestyle='--',
                        linewidth=1, label=f'Naive BS ({naive_bs:.3f})')
        plt.grid(True, alpha = 0.3)
        plt.xlabel('Normalized order lifetime')
        plt.ylabel('Brier score')
        plt.title(f'Brier score over order lifetime {model_name}')
        plt.legend()
        plt.show()
    
        print(heartbeat_metrics.to_string())
        return heartbeat_metrics

    eval_at_heartbeat(test_data, y_true, y_pred_prob, weights, dummy_fill_prob)

    
# =============================================================================
# 4. Performance Plot
# =============================================================================

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
    
    def safe_weigths(x:pd.DataFrame, y:str) -> float:
        if x['weights'].sum() == 0:
            return np.nan
        return np.average(x[y], weights = x['weights'])
    
    weighted_actual = grouped.apply(lambda x: safe_weigths(x, 'y_true')).dropna()
    weighted_pred = grouped.apply(lambda x: safe_weigths(x, 'y_pred')).dropna() 
    weighted_dummy = np.average(y_true, weights = weights)

    print(f'weighetd dummy {weighted_dummy}')
    
    plt.plot(weighted_pred, weighted_actual, color = 'b', label = 'Model')
    plt.plot(weighted_dummy,weighted_dummy, marker = 'o', markersize = 10, markeredgecolor = 'black', color = 'yellow', label = f'dummy {(weighted_dummy)*100:.2f}%')
    plt.plot([0,1], [0,1], color = 'black', label = 'Perfect')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Predicted Fill Probability')
    plt.ylabel('Actual Fill Probability')
    plt.grid(True, alpha = 0.3)
    plt.title(f'Performance of {model_name}')
    plt.legend()
    plt.show()
    
    plt.plot(weighted_pred, weighted_actual, color = 'b', label = 'Model')
    plt.plot(weighted_dummy,weighted_dummy, marker = 'o', markersize = 10, markeredgecolor = 'black', color = 'yellow', label = f'dummy {(weighted_dummy)*100:.2f}%')
    plt.plot([0,1], [0,1], color = 'black', label = 'Perfect')
    plt.xlim(0,.4)
    plt.ylim(0,.4)
    plt.xlabel('Predicted Fill Probability')
    plt.ylabel('Actual Fill Probability')
    plt.grid(True, alpha = 0.3)
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
    plt.plot(weighted_dummy, weighted_dummy, marker = 'o', markersize = 10, markeredgecolor = 'black', color = 'yellow', label = f'dummy {(weighted_dummy)*100:.2f}%')
    plt.plot([0,1], [0,1], color = 'black', label = 'Perfect')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Predicted Fill Probability')
    plt.grid(True, alpha = 0.3)
    plt.ylabel('Actual Fill Probability')
    plt.title(f'Performance of {model_name} On Orders only placed at Best Bid or Best Ask')
    plt.legend()
    plt.show()
    
        
    #Linegraph with on x axis the bins and on y axis the vol of orders landed in that bin 
    plt.figure(figsize = (20,10))
    vol_per_bin = df.groupby('bins', observed = False)['weights'].sum()
    middle = [b.mid for b in vol_per_bin.index]
    plt.plot(middle, vol_per_bin, color = 'b', label = 'LO Vol')
    plt.xlabel('Probability Bins')
    plt.xlim(0, 1)
    plt.ylabel('Log Vol of LOs')
    plt.grid(True, alpha = 0.3)
    plt.yscale('log')
    plt.title(f'Amount Of LO Vol appearing in each predictive probability bin {model_name}')
    plt.legend()
    plt.show()

        
    if model_name == "Light Gradient Boosted Model":
        plot_lgbm_importances(base_model, features)

    if model_name == 'FNN':
        plot_fnn_importances(calibrated_model, features, test_data, scalar)
        
# =============================================================================
# 5. Functions used in UserScript that return daily plots and metrics
# =============================================================================

def compute_daily_performance_curve(y_true:pd.Series, y_pred_prob:np.ndarray, weights:pd.Series, mask = None):
    
    """
    Calculates the performance curve for one single day, only used in userscript
    """
    
    if mask is not None:
        y_true = y_true[mask]
        y_pred_prob = y_pred_prob[mask]
        weights = weights[mask]
        
    df = pd.DataFrame({
        'y_true': y_true, 
        'y_pred': y_pred_prob, 
        'weights': weights
        })
    
    deltap = 0.01
    bins_low = np.arange(0, 0.401, deltap)
    bins_high = np.arange(0.43, 1, 3 * deltap)
    bins_custom = np.concatenate((bins_low, bins_high))

    df['bins'] = pd.cut(df['y_pred'], bins=bins_custom)
    grouped = df.groupby('bins', observed=False)
    
    def safe_weights(x:pd.DataFrame, col:str) -> float:
        if x['weights'].sum() == 0:
            return np.nan
        return np.average(x[col], weights=x['weights'])
    
    # Dont have dropna here like above since for the average we need same length arrays for every day, so empty bins are not dropped here
    weighted_actual = grouped.apply(lambda x: safe_weights(x, 'y_true'))
    weighted_pred = grouped.apply(lambda x: safe_weights(x, 'y_pred')) 
    
    vol_per_bin = grouped['weights'].sum()
    
    return weighted_pred.values, weighted_actual.values, vol_per_bin.values

def compute_daily_divergence(merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Function to calculate volume in bins based on regular and probq imbalance features, only used in userscript
    """

    merged['buy_vol'] = np.where(merged['BorS'] == -1, merged['Vol'],0)
    merged['vol'] = merged['Vol']


    bins = np.linspace(-1.0, 1.0,101)
    merged['Reg_Fine_Bin'] = pd.cut(merged['QImbalance'], bins=bins)
    merged['Prob_Fine_Bin'] = pd.cut(merged['ProbQImbal'], bins=bins)
    
    # Calculate the proportion of buys in each bin
 
    reg_buy_vol = merged.groupby('Reg_Fine_Bin', observed=False)['buy_vol'].sum()
    reg_total_vol = merged.groupby('Reg_Fine_Bin', observed=False)['vol'].sum()
    
    prob_buy_vol = merged.groupby('Prob_Fine_Bin', observed=False)['buy_vol'].sum()
    prob_total_vol = merged.groupby('Prob_Fine_Bin', observed=False)['vol'].sum()
    
    return reg_buy_vol.values, reg_total_vol.values, prob_buy_vol.values, prob_total_vol.values



def compute_daily_alligator(y_true: pd.Series, y_pred_prob: np.ndarray, weights: pd.Series, time_since_placement: pd.Series, order_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Construct daily plot which shows normalized lifetime on x-axis and predicted fill prob on y-axis with orders split analyzed based on whether they end up filling or not
    """


    eval_df = pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_pred_prob,
        'weight': weights.values,
        'tsp':    time_since_placement.values,
        'ID':     order_ids.values 
    })
    
    eval_df['EventualOutcome'] = eval_df['y_true']
    
    # Normalize each order's time axis to [0, 1]
    eval_df['NormTime'] = eval_df.groupby('ID')['tsp'].transform(
        lambda x: x / x.max() if x.max() > 0 else 0.0
    )
    
    # Create exactly 20 bins from 0 to 1
    bins = np.linspace(0, 1, 21) 
    eval_df['TimeBucket'] = pd.cut(eval_df['NormTime'], bins=bins, include_lowest=True)
    
    # Safe weight calculator to prevent ZeroDivisionError on empty bins
    def safe_alligator_weights(x: pd.DataFrame) -> float:
        if x['weight'].sum() == 0:
            return np.nan
        return np.average(x['y_pred'], weights=x['weight'])
    
    # observed=False ensures all 20 bins are kept even if some are empty for a specific day
    trajectory = (
        eval_df.groupby(['TimeBucket', 'EventualOutcome'], observed=False)
        .apply(safe_alligator_weights)
        .unstack() 
    )
    
    # Extract columns safely, defaulting to NaNs if a day magically had zero fills or zero cancels
    fills = trajectory[1].values if 1 in trajectory.columns else np.full(20, np.nan)
    cancels = trajectory[0].values if 0 in trajectory.columns else np.full(20, np.nan)
    
    return fills, cancels
    
def compute_daily_PR(y_true: pd.Series, y_pred_prob: np.ndarray, weights: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Computes precision, recall for data of a given trading day
    """
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob, sample_weight = weights)
    return precision, recall

def calc_weighted_ece(y_true: pd.Series, y_pred: np.ndarray, weights: pd.Series) -> float:
    
    """
    calcualtes the volumeweighted expected calibration error 
    """
    
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'weight': weights})
    
    deltap = 0.01
    bins_low = np.arange(0, 0.401, deltap)
    bins_high = np.arange(0.43, 1, 3 * deltap)
    bins = np.concatenate((bins_low, bins_high))
    n_bins = len(bins)
    df = df.sort_values('y_pred')
    df['bin'] = pd.qcut(df['y_pred'], q=n_bins, duplicates='drop')
    
    grouped = df.groupby('bin', observed=False)
    
    ece = 0.0
    total_weight = df['weight'].sum()
    
    for _, group in grouped:
        bin_weight = group['weight'].sum()
        if bin_weight > 0:

            actual_rate = np.average(group['y_true'], weights=group['weight'])
            pred_rate = np.average(group['y_pred'], weights=group['weight'])

            ece += (bin_weight / total_weight) * np.abs(actual_rate - pred_rate)
            
    return ece

def compute_daily_scores(y_true: pd.Series, y_pred_prob: np.ndarray, weights: pd.Series) -> dict[str, float]:
    """
    #compute brier logloss and skill scores daily to be used in walk forward in userscript to get average model results
    """
    
    dummy_fill_prob = np.average(y_true, weights = weights)
    brier_score = brier_score_loss(y_true, y_pred_prob, sample_weight = weights)
    brier_skill_score = 1 - ((brier_score)/(dummy_fill_prob*(1-dummy_fill_prob)))
    logloss_score = log_loss(y_true, y_pred_prob, sample_weight = weights)
    
    dummy_y_pred_prob = np.full(len(y_true), dummy_fill_prob) 
    dummy_logloss = log_loss(y_true, dummy_y_pred_prob, sample_weight = weights)
    
    avg_precision = average_precision_score(y_true, y_pred_prob, sample_weight = weights)
    
    logloss_skill_score = 1 - ((logloss_score)/(dummy_logloss))
    
    dummy_brierscore = brier_score_loss(y_true, dummy_y_pred_prob, sample_weight = weights)
    
    avgprecision_dummy = average_precision_score(y_true, dummy_y_pred_prob, sample_weight = weights)
    
    ece_score = calc_weighted_ece(y_true, y_pred_prob, weights)
    pr_ratio = avg_precision / avgprecision_dummy
    
    scores = {
        
        'brier': brier_score,
        'brier_skill': brier_skill_score,
        'logloss': logloss_score,
        'logloss_skill': logloss_skill_score,
        'pr': avg_precision,
        'ece': ece_score,
        'pr_ratio': pr_ratio,
        
        'dummy_fill_prob': dummy_fill_prob,
        'dummy_brier': dummy_brierscore,
        'dummy_logloss': dummy_logloss,
        'dummy_pr': avgprecision_dummy
        
        }
    
    return scores