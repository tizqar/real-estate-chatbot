# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 22:54:20 2025

@author: milos
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap # Added import for SHAP

def load_and_preprocess_data(train_path='./train.csv', test_path='./test.csv'):
    """
    Loads training and testing data, performs comprehensive preprocessing,
    and returns processed training and testing DataFrames with only specified features.
    """
    df_train_orig = pd.read_csv(train_path)
    df_test_orig = pd.read_csv(test_path)

    df_train_orig['is_train'] = True
    df_test_orig['is_train'] = False

    test_pids = df_test_orig['PID']

    combined_df = pd.concat([df_train_orig.drop('PID', axis=1), df_test_orig.drop('PID', axis=1)], ignore_index=True)

    # Log-transforming the sale price for the training portion to handle skewness
    if 'SalePrice' in combined_df.columns:
        combined_df.loc[combined_df['is_train'], 'SalePrice'] = np.log1p(combined_df.loc[combined_df['is_train'], 'SalePrice'])

    if 'SalePrice' in combined_df.columns:
        outlier_mask = (combined_df['is_train']) & \
                       (combined_df['Gr Liv Area'] > 4000) & \
                       (combined_df['SalePrice'] < np.log1p(300000))
        combined_df = combined_df[~outlier_mask].reset_index(drop=True)


    #Handling Missing Values
    for col in ('Alley', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1',
                'BsmtFin Type 2', 'Fence', 'Fireplace Qu', 'Garage Type', 'Garage Finish',
                'Garage Qual', 'Garage Cond', 'Mas Vnr Type', 'Misc Feature', 'Pool QC'):
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna('None')

    for col in ('Garage Yr Blt', 'Garage Area', 'Garage Cars', 'BsmtFin SF 1', 'BsmtFin SF 2',
                'Bsmt Unf SF', 'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Mas Vnr Area'):
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna(0)

    if 'MS Zoning' in combined_df.columns and combined_df['MS Zoning'].isnull().any():
        combined_df['MS Zoning'] = combined_df.groupby('MS SubClass')['MS Zoning'].transform(lambda x: x.fillna(x.mode()[0]))
    for col in ['Electrical', 'Kitchen Qual', 'Exterior 1st', 'Exterior 2nd', 'Functional', 'Sale Type']:
        if col in combined_df.columns and combined_df[col].isnull().any():
            combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

    for col in combined_df.select_dtypes(include=np.number).columns:
        if combined_df[col].isnull().sum() > 0:
            combined_df[col] = combined_df[col].fillna(combined_df[col].median())

    #Feature Creation and Transformation
    combined_df['Age'] = combined_df['Yr Sold'] - combined_df['Year Built']
    combined_df['Total SF'] = combined_df['Total Bsmt SF'] + combined_df['1st Flr SF'] + combined_df['2nd Flr SF']
    combined_df['Total Baths'] = combined_df['Full Bath'] + (0.5 * combined_df['Half Bath']) + combined_df['Bsmt Full Bath'] + (0.5 * combined_df['Bsmt Half Bath'])
    combined_df['Total Porch SF'] = combined_df['Open Porch SF'] + combined_df['3Ssn Porch'] + combined_df['Enclosed Porch'] + combined_df['Screen Porch'] + combined_df['Wood Deck SF']
    combined_df['Has Pool'] = combined_df['Pool Area'].apply(lambda x: 1 if x > 0 else 0)
    combined_df['Has Garage'] = combined_df['Garage Area'].apply(lambda x: 1 if x > 0 else 0)

    # Ordinal Feature Mapping
    qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    for col in ['Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Heating QC', 'Kitchen Qual', 'Fireplace Qu', 'Garage Qual', 'Garage Cond', 'Pool QC']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].map(qual_map).fillna(0) # Fillna(0) for any not-in-map values

    selected_features = [
        #'Age',
        'Overall Qual',
        'Gr Liv Area',
        'Lot Area',
        'Kitchen Qual',    
        'Overall Cond', 
        'Exter Qual',       
        'Year Built',              
        'Total SF',          
        'Total Baths'
    ]

    features_to_keep = selected_features.copy()
    if 'SalePrice' in combined_df.columns:
        features_to_keep.append('SalePrice')
    features_to_keep.append('is_train')

    df_filtered = combined_df[features_to_keep].copy()

    # Split back into final training and testing dataframes
    final_train = df_filtered[df_filtered['is_train']].drop('is_train', axis=1)
    final_test = df_filtered[~df_filtered['is_train']].drop(['is_train', 'SalePrice'], axis=1, errors='ignore')

    final_train.to_csv('final_preprocessed_data.csv', index=False)

    return final_train, final_test, test_pids


def build_pipeline(model):
    """
    Builds a scikit-learn pipeline for the given model, including preprocessing steps.
    Currently, it applies StandardScaler to all numerical features.
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=np.number))
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


def train_and_evaluate(df):
    """
    -trains an XGBoost regression model,
    -evaluates its performance,
    -generates SHAP plots for feature importance.

    Args:
        df (pd.DataFrame)

    Returns:
        tuple: A tuple containing:
            - results (dict): A dictionary with performance metrics (MSE, RMSE, R2) for the XGBoost model.
            - trained_pipeline: The trained pipeline of the XGBoost model.
            - X_val (pd.DataFrame): Features of the validation set.
            - val_y_orig (pd.Series): Actual target values for the validation set, inverse-transformed.
            - val_preds_orig (np.array): Predictions on the validation set, inverse-transformed.
    """

    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBRegressor(
        max_depth=4,           
        n_estimators=200,      
        colsample_bytree=0.6,   # Subsample ratio of columns when constructing each tree
        learning_rate=0.05,     
        random_state=42,       
        subsample=0.5,          
        min_child_weight=5,     # Minimum sum of instance weight (hessian) needed in a child
        colsample_bylevel=0.7,  # Subsample ratio of columns for each level
        n_jobs=-1        
    )

    model_name = 'XGBoost'
    trained_pipeline = build_pipeline(xgb_model)
    trained_pipeline.fit(X_train, y_train)

    preds_log = trained_pipeline.predict(X_val)

    val_y_orig = np.expm1(y_val)
    val_preds_orig = np.expm1(preds_log)

    mse = mean_squared_error(y_val, preds_log)
    rmse = np.sqrt(mean_squared_error(val_y_orig, val_preds_orig))
    r2 = r2_score(y_val, preds_log)

    results = {model_name: {'MSE': mse, 'RMSE': rmse, 'R2': r2}}
    print("\n--- Model Performance ---")
    print(f"{model_name}: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    print("\n--- Summary ---")
    print(f"The trained model is: {model_name} with an RMSE of {rmse:.4f}")

    print("\n--- Generating SHAP Feature Importance Plots ---")

    preprocessor = trained_pipeline.named_steps['preprocessor']
    xgb_regressor_model = trained_pipeline.named_steps['model']

    X_val_processed = preprocessor.transform(X_val)

    #DataFrame with processed features and original column names is crucial for SHAP
    feature_names = X.columns.tolist()
    X_val_processed_df = pd.DataFrame(X_val_processed, columns=feature_names, index=X_val.index)

    explainer = shap.TreeExplainer(xgb_regressor_model)
    shap_values = explainer.shap_values(X_val_processed_df)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val_processed_df, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (Bar Plot)", fontsize=16)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val_processed_df, show=False)
    plt.title("SHAP Detailed Feature Importance (Beeswarm Plot)", fontsize=16)
    plt.tight_layout()
    plt.show()

    return results, trained_pipeline, model_name, X_val, val_y_orig, val_preds_orig

def plot_predictions_vs_actuals(y_true, y_pred, model_name="Model"):
    """
    Creates a scatter plot comparing predicted values against actual values.
    Includes an ideal 'y=x' line for visual comparison.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, s=100, edgecolor='black', linewidth=0.5)
    # Add a diagonal line representing perfect predictions
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
    plt.title(f'Predicted vs. Actual Prices for {model_name} (Validation Set)', fontsize=16)
    plt.xlabel('Actual Sale Price ($)', fontsize=12)
    plt.ylabel('Predicted Sale Price ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, model_name="Model"):
    """
    Creates a scatter plot of residuals against predicted values..
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, s=100, edgecolor='black', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residuals') # Horizontal line at y=0
    plt.title(f'Residuals Plot for {model_name} (Validation Set)', fontsize=16)
    plt.xlabel('Predicted Sale Price ($)', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted) ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 1. Load and Preprocess Data
    print("Starting data loading and preprocessing...")
    train_df, test_df, test_pids = load_and_preprocess_data()
    print("Data preprocessing complete.")
    print(f"Processed training data shape: {train_df.shape}")
    print(f"Processed test data shape: {test_df.shape}")

    # 2. Train and Evaluate the XGBoost Model
    print("\nStarting XGBoost model training and evaluation...")
    results, trained_model_pipeline, model_name, X_val, val_y_orig, val_preds_orig = train_and_evaluate(train_df)
    print(f"Model training and evaluation complete for {model_name}.")

    # 3. Display XGBoost Specific Results Summary
    print("\n" + "="*40)
    print("          XGBoost Model Performance Summary")
    print("="*40)
    if model_name in results:
        xgb_results = results[model_name]
        print(f"  MSE (on log-transformed prices): {xgb_results['MSE']:.4f}")
        print(f"  RMSE (on original price scale):  {xgb_results['RMSE']:.4f}")
        print(f"  R-squared ($R^2$):               {xgb_results['R2']:.4f}")
    else:
        print("  XGBoost results could not be found.")
    print("="*40 + "\n")

    # Data Analysis and Charts
    print("\n--- Generating Performance Charts (Predicted vs Actual, Residuals) ---")
    plot_predictions_vs_actuals(val_y_orig, val_preds_orig, model_name=model_name)
    plot_residuals(val_y_orig, val_preds_orig, model_name=model_name)

    # 4. Make Predictions on the Unseen Test Set
    if trained_model_pipeline:
        print(f"\nMaking predictions on the *unseen* test data using the {model_name} model...")
        test_predictions_log = trained_model_pipeline.predict(test_df)
        final_predictions_original_scale = np.expm1(test_predictions_log)
        print("Sample predictions on unseen test data (original scale):")
        for i, pred in enumerate(final_predictions_original_scale[:5]):
            print(f"  Test Sample {i+1}: ${pred:,.2f}")

        print("\n" + "="*40)
        print("      Validation Set: Sample Predicted vs. Actual Prices")
        print("      (from train-test split for model evaluation)")
        print("="*40)
        if val_preds_orig is not None and val_y_orig is not None:
            num_samples_to_display = min(10, len(val_preds_orig))
            for i in range(num_samples_to_display):
                print(f"  Sample {i+1}: Predicted: ${val_preds_orig[i]:,.2f}, Actual: ${val_y_orig.iloc[i]:,.2f}")
        else:
            print("  Validation set predictions or actuals not available.")
        print("="*40 + "\n")
    else:
        print("No trained model pipeline was returned. Cannot make test predictions or display validation results.")

    # 5. Save the trained model pipeline
    try:
        with open('model1.pkl', 'wb') as file:
            pickle.dump(trained_model_pipeline, file)
        print("\nModel has been saved as 'model.pkl'")
    except Exception as e:
        print(f"Error saving model: {e}")
        