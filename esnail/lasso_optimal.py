import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


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

# Lasso with found optimal hyperparameter values
lasso = Lasso(
    alpha=0.0060452910025173895,
    selection='random',
    max_iter=10000,
    random_state=42
)

lasso.fit(X_train_scaled, y_train)

# Check to make sure results are same
predictions = lasso.predict(X_test_scaled)

new_r2 = r2_score(y_test, predictions)
new_rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "="*50)
print(f"Tuned Lasso Regression Model Results")
print(f"   R2 Score: {new_r2:.4f}")
print(f"   RMSE:     {new_rmse:.4f} μV/K")
print("="*50)

# Find most important features (Lasso has no feature_importances_, so use the
# absolute value of its coefficients instead)
importances = np.abs(lasso.coef_)

# Create data frame ranking the most important features
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances,
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

# Display top 20 features
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df["Feature"][:20][::-1], feature_importance_df["Importance"][:20][::-1])
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlabel("|Lasso Coefficient|")
plt.title(f"Top 20 Most Important Features (Tuned Lasso)")
plt.tight_layout()
plt.show()

print("\nTop 5 absolute best predictors:")
print(feature_importance_df.head(5))
