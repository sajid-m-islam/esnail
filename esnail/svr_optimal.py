import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance


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

# SVR with found optimal hyperparameter values
svr = SVR(
    kernel='rbf',
    C=98.49492058649535,
    epsilon=0.6498734863562683,
    gamma=0.8335370678166605,
    max_iter=50000
)

svr.fit(X_train_scaled, y_train)

# Check to make sure results are same
predictions = svr.predict(X_test_scaled)

new_r2 = r2_score(y_test, predictions)
new_rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "="*50)
print(f"Tuned Support Vector Regressor Results")
print(f"   R2 Score: {new_r2:.4f}")
print(f"   RMSE:     {new_rmse:.4f} μV/K")
print("="*50)

# Find most important features (SVR has no coef_/feature_importances_ for a
# non-linear kernel, so use permutation importance on the held-out test set instead)
perm_result = permutation_importance(svr, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importances = perm_result.importances_mean

# Create data frame ranking the most important features
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances,
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

# Display top 20 features
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df["Feature"][:20][::-1], feature_importance_df["Importance"][:20][::-1])
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlabel("Permutation Importance (drop in R² when shuffled)")
plt.title(f"Top 20 Most Important Features (Tuned SVR)")
plt.tight_layout()
plt.show()

print("\nTop 5 absolute best predictors:")
print(feature_importance_df.head(5))

# Create graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Tuned SVR Performance', fontsize=16)

# # Plot 1: Predicted vs Actual
# ax1.scatter(y_test, predictions, alpha=0.5, color='teal')
# min_val = min(y_test.min(), predictions.min())
# max_val = max(y_test.max(), predictions.max())
# ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
# ax1.set_title('Predicted vs. Actual')
# ax1.set_xlabel('True Seebeck Coefficient')
# ax1.set_ylabel('Predicted Seebeck Coefficient')

# # Plot 2: Residuals
# residuals = y_test - predictions
# ax2.scatter(predictions, residuals, alpha=0.5, color='coral')
# ax2.axhline(0, color='black', linestyle='--', lw=2)
# ax2.set_title('Residual Distribution')
# ax2.set_xlabel('Predicted Seebeck Coefficient')
# ax2.set_ylabel('Error / Residual')

# plt.tight_layout()
# plt.show()
