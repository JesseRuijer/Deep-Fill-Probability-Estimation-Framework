# Deep Fill Probability Estimation Framework

We propose a framework that computes instantaneous limit order fill probabilities that conditionally update
using various machine- and deep-learning models based on level 3 Nasdaq ITCH limit order book data. The
use of carefully constructed feature engineering, with the notable addition of a ’heartbeat mechanic’ to infer
time-related features as well as highly optimized code, allows for clear analysis of model decision-making and
high efficiency. Proper scoring rules such as weighted log-loss and custom weighting are employed to grade
model performance and comparisons to the baseline, which our models significantly outperform. Finally, we
propose a fill-probability weighted alternative to the traditional queue imbalance to enhance predictive signal
strength. This repository contains all the code used in this project (when paper is published link to paper will be added).

## Project Structure

### 1. Data Processing & Feature Engineering
*   `DataAndFeatureEngineering.py`: Data importing, cleaning, feature extraction, heartbeat engine
*   `FileManager.py`: Handles the data entries into the programme, codes a cache_file for easy reruns
*   `ExploratoryData.py`: Exploratory analysis for when importing new asset or trading day file

### 2. Machine Learning Engines
*   `LightGBMEngine.py`: LGBM Engine. Trains & Calibrates
*   `LogisticRegressionEngine.py`: LR Engine. Trains & Calibrates
*   `FNN.py`: Feedforward Neural Net Training. Wrapper class to convert outputs to comply with ModelEvaluation function input requirements
*   `HyperParOptimiser[...].py`: Script to use Optuna to find best hyperparameters for [...]

### 3. Evaluation & Inference
*   `UserScript.py`: Allows for the majority of the functionality of the framework to be easily accessed by the user. Contains for all models: training, testing, using, evaluation 
*   `ModelEvaluation.py`: Creates all model evaluation plots as well as calculation of evaluation metrics
*   `FeatureFinder.py`: Script to find relevant features for model, give this script all features, and it returns important ones

### 4. Setup / Additional
*   `config.py`: Configuration script, includes all the constant variables as well as lists of variables used in each model
*   `Functions.py`: Contains helper functions that are used throughout the framework
*   `Main.py`: Main script for controlling the process, to access functionality, go to UserScript
  
## Data Prerequisites & Structure

Due to licensing and size constraints, the raw Level 3 NASDAQ Limit Order Book data is not included in this repository. To run this pipeline, you must provide your own MATLAB (`.mat`) data files formatted to match the expected schema.

The automated file manager (`FileManager.py`) expects raw data to be organized by ticker and strictly named using the `TICKER_YYYYMMDD_NASDAQ` format. 

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

```

## Requirements

* pandas
* numpy
* scipy
* matplotlib
* seaborn
* torch
* shap
* scikit-learn
* joblib
* optuna
* lightgbm

## How to use

* Install dependencies 
* Install required data
* Run UserScript file (CLI should be explanatory here)

## Contact & Citation
If you are seriously interested in reviewing this architecture, you can reach out to me via below. **A sample of the dataset used to validate this pipeline can be provided upon request for academic review.** Also please consider citing this work if used in any way.

* **Name:** Jesse Ruijer
* **Email:** jesse.ruijer@gmail.com
* **Academic Affiliation:** Joint MSc Quantitative Finance, ETH Zurich / University of Zurich
* **LinkedIn:** https://www.linkedin.com/in/jesse-ruijer-898b70281/

*(Citation link to the published paper will be added here upon publication).*

