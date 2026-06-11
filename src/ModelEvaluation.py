#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:17:44 2026

@author: jesseruijer
"""


def test_model():
    
    X_test = regressormatrix2[log_mdl_features]
    X_test_lgbm = regressormatrix2[lgbm_mdl_features]
    
    X_test_scaled = scalar.transform(X_test) #Here we transform and not fit anymore i.e we use same scale as above so comparisons are valid
y_true = regressormatrix2["Fill_NoFill"]
 #Evaluate performance metrics 
print("Light GBM Engine metrics")
 
brierscore_lgbm = brier_score_loss(y_true, y_pred_prob_lgbm)
print(f'Brier score is {brierscore_lgbm:.3f}')
 
logloss_lgbm = log_loss(y_true, y_pred_prob_lgbm)
print(f'Logloss score is {logloss_lgbm:.3f}')
 
#However i think AUC is only reliable on balanced data which this totally isnt, so the AUC is artificially inflated, why because we rarely ever have a fill and AUC is area under ROC, ROC formula is 
 
aucscore_lgbm = roc_auc_score(y_true, y_pred_prob_lgbm)
print(f'AUC score is {aucscore_lgbm:.3f}')
 
avgprecision_lgbm = average_precision_score(y_true, y_pred_prob_lgbm)
print(f'Avg precision score is {avgprecision_lgbm:.3f}')

y_pred_prob_lgbm = calibrated_lgbm.predict_proba(X_test_lgbm)[:, 1]




#Do some prediction using scikit learn
y_pred = calibrated_model.predict(X_test_scaled)
y_pred_prob = calibrated_model.predict_proba(X_test_scaled)[: , 1] # We are now only looking at the fill probabilities

model_coef_df = pd.DataFrame(
    
    {
     "Feature": log_mdl_features,
     "Coefficient (Log Odds)" : base_lr_model.coef_[0],  # We only predict binary classification so our model only has one row so access that with [0]
     "Odds Ratio": np.exp(base_lr_model.coef_[0])
     }
    
    )
print(f'                  Logistic Regression ORs \n {model_coef_df.sort_values(by = "Odds Ratio", key=abs, ascending=False)}')


#Baseline fill percentage which i defined as the number of ones divided by number of ones and zeros in fill_map, which guarantees uniqueness by the fact i used .last in code before it
print(f"Baseline Fill percentage is {regressormatrix2['Fill_NoFill'].mean()*100:.2f} %")

#because were making a probability engine using logistic regression we must look at brier score and log loss to evaulte it
#Confusion matrices and precisions for ex dont make too much sense here since then you need to define a treshold for when a probability gets put in the category
#0 or 1 where for us that doesnt matter we're just interested in the pure probability of an order beig filled

print("Engine metrics")

brierscore = brier_score_loss(y_true, y_pred_prob)
print(f'Brier score is {brierscore:.3f}')

logloss = log_loss(y_true, y_pred_prob)
print(f'Logloss score is {logloss:.3f}')

#However i think AUC is only reliable on balanced data which this totally isnt, so the AUC is artificially inflated, why because we rarely ever have a fill and AUC is area under ROC, ROC formula is 

aucscore = roc_auc_score(y_true, y_pred_prob)
print(f'AUC score is {aucscore:.3f}')


avgprecision = average_precision_score(y_true, y_pred_prob)
print(f'Avg precision score is {avgprecision:.3f}')







#Visualisiton of performance and comparison to baseline dummy model which just guesses a baseline percentage on each order for it being filled 
#Dummy y fill prob is just an array of length y true with all entries equal to dummy fill prob

dummy_fill_prob = regressormatrix['Fill_NoFill'].mean()
dummy_y_pred_prob = np.full(len(y_true), dummy_fill_prob) #just creates an array of length y true with dummy fill probs


print("Dummy metrics")

dummy_brierscore = brier_score_loss(y_true, dummy_y_pred_prob)
print(f'Dummy Brier score is {dummy_brierscore:.3f}')

dummy_logloss = log_loss(y_true, dummy_y_pred_prob)
print(f' Dummy Logloss score is {dummy_logloss:.3f}')

dummy_aucscore = roc_auc_score(y_true, dummy_y_pred_prob)
print(f'Dummy AUC score is {dummy_aucscore:.3f}')

avgprecision_dummy = average_precision_score(y_true, dummy_y_pred_prob)
print(f'Avg precision score is {avgprecision_dummy:.3f}')

#Visualisation of performance vs dummy
fig, axes = plt.subplots(1,3, figsize = (24,8))
#calibration curve
engine_true, engine_prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10, strategy = 'quantile') #tuple unpacking since the function returns two variables, we just name them immediately in one line
axes[0].plot([0,1], [0,1], color = 'grey', label = "Perfect Calibration") # axes[0] means we're talking about the left figure then [0,1] , [0,1] are x list and y list and are read vertically so the first point is 0,0 and the second point is 1,1 and a line is drawn between them i.e the perfect prediction line i think but check this
axes[0].plot(engine_prob_pred, engine_true, color = 'b' ,label = 'Logistic Regression Engine')
axes[0].set_title('Calibration curve')
axes[0].legend()

#Roc curve

engine_fpr, engine_tpr, tresholds = roc_curve(y_true, y_pred_prob) #returns false postive rates and true positive rates, treshold which i think is the number or prob above or below it gives a certain classification
dummy_fpr, dummy_tpr, tresholds = roc_curve(y_true, dummy_y_pred_prob)
axes[1].plot(engine_fpr, engine_tpr, color = 'b', label = 'Logistic Regression Engine')
axes[1].plot(dummy_fpr, dummy_tpr, color = 'r', label = 'Dummy')
axes[1].legend()

#Precision recall curve
engine_precision, engine_recall, engine_treshold = precision_recall_curve(y_true, y_pred_prob)
axes[2].plot(engine_recall, engine_precision, color = 'b', label = 'Engine PR')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].plot([0,1], [dummy_fill_prob, dummy_fill_prob], color = 'red', label = 'Dummy') #the dummy PR is just the baseline fill rate i.e here just a horizontal line

plt.show()


def predict_order_fill_prob(features):
    #predicts specific probability for a given limit order being filled using the logistic regression engine from above
    #since pd dfs are slow its better to use np array here
    
    input_array = np.array(features).reshape(1,-1) #resshape needed 1 means passing 1 row, -1 means calculate the right dim for the columns so this creates a matrix which is whats needed for sci kit later
    scaled_input  = (input_array - scalar.mean_) / scalar.scale_ # I think the trailing _ tells sci kit to look at the fitted values and calculate mean and std of those                   using scalar so we do the standardizations for each value and not all at the same time
    fill_prob = calibrated_model.predict_proba(scaled_input)[0,1]
    
    return fill_prob

example_state = X_test.iloc[67].values # Expects numerical list with values corresponding to the entries above
print(predict_order_fill_prob(example_state))
print(X_test.iloc[67])
#Better to also create a function that extracts these features for maybe a given order ID?


    
    #Plots Correlation heat plot of all features and of features that appear in model
    
    features_all = ['BASpread', 'QImbalance', 'AbsQImbalance', 'TotalVolImbalance', 'Weighted Vol Imbalance', 
                'Midprice', 'Microprice', "DistanceToTouch", 'LogVolAhead', "LookBackHiddenVol"] 
    
    features_in_model = ['AbsQImbalance', 'Weighted Vol Imbalance', 'Microprice', 
                         "DistanceToTouch", 'LogVolAhead', "LookBackHiddenVol"] 
    corr_matrix = regressormatrix[features_all].corr()
    corr_matrix2 = regressormatrix[features_in_model].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("All Feature Correlation Matrix")
    plt.show()
    
    
    sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix Of Features In Model")
    plt.show()