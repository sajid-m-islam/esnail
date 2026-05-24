import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor



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

# Define hyperparameter grid
param_grid = {
    'n_estimators': [300],      # Number of trees in the forest
    'max_depth': [10, 12, 15, 20],       # Maximum depth of the tree
    'min_samples_split': [4, 5, 6],       # Min samples required to split a node
    'min_samples_leaf': [1, 2],          # Min samples required at each leaf
    'max_features': [1.0, 0.8],         # How much of each feature to use
    'bootstrap': [True],
    'max_samples': [0.75, 0.90, None]    # How much data to use
}

rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1,scoring='neg_mean_squared_error')

rf_grid.fit(X_train_scaled, y_train)

best_rf = rf_grid.best_estimator_
print("\nTuning Complete. The optimal parameters found are:")
for param, value in rf_grid.best_params_.items():
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