import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import optuna



# Setup
df = pd.read_csv("./data/reduced_features_v1.csv")

target_col = 'Seebeck Coefficient'

cols_to_drop = ['Composition', 'Site_X', 'Site_Y', 'Site_Z', 
                'Seebeck Coefficient']

X = df.drop(columns=cols_to_drop)
y = df[target_col]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter search space and Optuna objective
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),      # Number of trees in the forest
        'max_depth': trial.suggest_int('max_depth', 10, 20),             # Maximum depth of the tree
        'min_samples_split': trial.suggest_int('min_samples_split', 4, 6),   # Min samples required to split a node
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 2),     # Min samples required at each leaf
        'max_features': trial.suggest_float('max_features', 0.8, 1.0),       # How much of each feature to use
        'bootstrap': True,
        'max_samples': trial.suggest_float('max_samples', 0.75, 1.0)         # How much data to use
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

best_rf = RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=True, **study.best_params)
best_rf.fit(X_train_scaled, y_train)

print("\nTuning Complete. The optimal parameters found are:")
for param, value in study.best_params.items():
    print(f"   -> {param}: {value}")
    
print("\nEvaluating the Tuned Model on the hidden Test Set")
predictions = best_rf.predict(X_test_scaled)

new_r2 = r2_score(y_test, predictions)
new_rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "="*50)
print(f"Tuned Random Forest Model Results")
print(f"   R2 Score: {new_r2:.4f}")
print(f"   RMSE:     {new_rmse:.4f} μV/K")
print("="*50)

# Create graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Tuned Random Forest Performance', fontsize=16)

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