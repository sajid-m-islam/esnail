import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb


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

# Initialize baseline models
models = {
    "Lasso Regression": Lasso(random_state=42),
    "Support Vector Regressor": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
}

# Compute each model results
results = []
predictions_dict = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    predictions_dict[name] = predictions

    
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    results.append({
        "Model": name,
        "R2 Score": r2,
        "RMSE": rmse
    })

    print(f"Model {name} trained.")

# Print summary
print("\n" + "="*50)
print("Final Ranking")
print("="*50)

# Convert results to a DataFrame and sort by R2 Score (highest is best)
results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))
print("="*50)

# Create visuals

# Plot 1: Predicted vs. Actual
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Predicted vs. Actual Seebeck Coefficient', fontsize=16)

for ax, (name, preds) in zip(axes.flatten(), predictions_dict.items()):
    ax.scatter(y_test, preds, alpha=0.5, color='teal')
    
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_title(name)
    ax.set_xlabel('True Seebeck Coefficient')
    ax.set_ylabel('Predicted Seebeck Coefficient')
    
plt.tight_layout()
plt.show()


# Plot 2: Residuals Plot (Prediction Errors)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Residual Distribution (True - Predicted)', fontsize=16)

for ax, (name, preds) in zip(axes.flatten(), predictions_dict.items()):
    residuals = y_test - preds
    ax.scatter(preds, residuals, alpha=0.5, color='coral')
    ax.axhline(0, color='black', linestyle='--', lw=2)
    
    ax.set_title(name)
    ax.set_xlabel('Predicted Seebeck Coefficient')
    ax.set_ylabel('Error / Residual')
    
plt.tight_layout()
plt.show()


# Plot 3: Metrics Bar Chart
fig, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df['Model']))
width = 0.35

rects1 = ax1.bar(x - width/2, results_df['R2 Score'], width, label='R² Score', color='teal')
ax1.set_ylabel('R² Score', color='teal', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='teal')

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, results_df['RMSE'], width, label='RMSE', color='coral')
ax2.set_ylabel('RMSE', color='coral', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='coral')

ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Model'], rotation=15)
plt.title('Model Performance Comparison', fontweight='bold')

fig.tight_layout()
plt.show()
