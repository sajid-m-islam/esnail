import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
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

# Clip extreme outliers to the training distribution's range before scaling.
# Lasso's prediction (sum of coefficient * scaled value) has no ceiling, so a
# leverage outlier (a feature value far outside anything seen in training) can
# blow up to an enormous prediction no matter which scaler is used. Tree/kernel
# models are naturally bounded by the training data's range; Lasso needs this
# bound enforced explicitly. Bounds are derived from the training set only and
# applied identically to train and test to avoid leakage.
clip_lower = X_train.quantile(0.01)
clip_upper = X_train.quantile(0.99)
X_train = X_train.clip(lower=clip_lower, upper=clip_upper, axis=1)
X_test = X_test.clip(lower=clip_lower, upper=clip_upper, axis=1)

# Scale data
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter search space and Optuna objective
def objective(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),      # Regularization strength
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),  # Coefficient update order
        'max_iter': 10000,  # High enough to let alpha/selection actually converge
    }
    lasso = Lasso(random_state=42, **params)
    scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

best_lasso = Lasso(random_state=42, max_iter=10000, **study.best_params)
best_lasso.fit(X_train_scaled, y_train)

print("\nTuning Complete. The optimal parameters found are:")
for param, value in study.best_params.items():
    print(f"   -> {param}: {value}")

print("\nEvaluating the Tuned Model on the hidden Test Set")
predictions = best_lasso.predict(X_test_scaled)

new_r2 = r2_score(y_test, predictions)
new_rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "="*50)
print(f"Tuned Lasso Regression Model Results")
print(f"   R2 Score: {new_r2:.4f}")
print(f"   RMSE:     {new_rmse:.4f} μV/K")
print("="*50)

# Create graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Tuned Lasso Regression Performance', fontsize=16)

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
