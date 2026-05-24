import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

def main():
        
    # Setup
    df = pd.read_csv("./data/heusler_reduced_features_NEW.csv")

    target_col = 'Seebeck Coefficient'

    cols_to_drop = ['Composition', 'Site_X', 'Site_Y', 'Site_Z', 
                    'Seebeck Coefficient', 'Electrical Resistivity']

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create parameter distribution
    param_grid = {
        'n_estimators': [100, 150],           # Number of sequential trees
        'learning_rate': [0.04, 0.05, 0.06],        # Step size shrinking to prevent overfitting
        'max_depth': [6, 7, 8],                      # XGBoost likes shallow trees!
        'min_child_weight': [4, 5, 6],               # Minimum sum of weights needed in a child (stops outlier hunting)
        'gamma': [4, 5, 6],                   # Minimum loss reduction to split a node (forces conservative learning)
        'subsample': [0.75, 0.8, 0.85],                   # Fraction of data to use per tree
        'colsample_bytree': [0.5, 0.6, 0.7]             # Fraction of features to use per tree
    }

    # Run xgboost with grid search to find best set of parameters
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    xgb_random = GridSearchCV(estimator=xg_reg, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1, scoring='neg_mean_squared_error')

    xgb_random.fit(X_train_scaled, y_train)

    best_xgb = xgb_random.best_estimator_
    print("\nTuning Complete. The optimal parameters found are:")
    for param, value in xgb_random.best_params_.items():
        print(f"   -> {param}: {value}")
        
    # Run xgboost with optimal parameters
    print("\nEvaluating the Tuned XGBoost on the strictly hidden Test Set")
    predictions = best_xgb.predict(X_test_scaled)

    new_r2 = r2_score(y_test, predictions)
    new_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n" + "="*50)
    print(f" GRID-TUNED XGBOOST RESULTS ")
    print(f"   R2 Score: {new_r2:.4f}")
    print(f"   RMSE:     {new_rmse:.4f} μV/K")
    print("="*50)

    # Create graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Tuned XGBoost Performance', fontsize=16)

    # Plot 1: Predicted vs Actual
    ax1.scatter(y_test, predictions, alpha=0.5, color='teal')
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_title('Predicted vs. Actual')
    ax1.set_xlabel('True Seebeck Coefficient')
    ax1.set_ylabel('Predicted Seebeck Coefficient')

    # Plot 2: Residuals
    residuals = y_test - predictions
    ax2.scatter(predictions, residuals, alpha=0.5, color='coral')
    ax2.axhline(0, color='black', linestyle='--', lw=2)
    ax2.set_title('Residual Distribution')
    ax2.set_xlabel('Predicted Seebeck Coefficient')
    ax2.set_ylabel('Error / Residual')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()