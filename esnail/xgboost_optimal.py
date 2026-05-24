import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


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

# Random forest with found optimal hyperparameter values
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=7,             
    learning_rate=0.04,
    gamma=5,
    min_child_weight=5,      
    subsample=0.85,          
    colsample_bytree=0.6,    
    random_state=42, 
    n_jobs=-1
)

xgb_reg.fit(X_train_scaled, y_train)

# Check to make sure results are same
predictions = xgb_reg.predict(X_test_scaled)

new_r2 = r2_score(y_test, predictions)
new_rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "="*50)
print(f"Tuned Random Forest Model Results")
print(f"   R2 Score: {new_r2:.4f}")
print(f"   RMSE:     {new_rmse:.4f} μV/K")
print("="*50)

# Find most important features
importances = xgb_reg.feature_importances_

# Create data frame ranking the most important features
feature_importance_df = pd.DataFrame({ 
    "Feature": X.columns, 
    "Importance": importances,
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

# Display top 20 features
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df["Feature"][:20][::-1], feature_importance_df["Importance"][:20][::-1])
plt.grid(axis='x', linestyle='--', alpha=0.7)    
plt.xlabel("XGBoost Importance Score")
plt.title(f"Top 20 Most Important Features (Tuned XGBoost)")
plt.tight_layout()
plt.show()

print("\nTop 5 absolute best predictors:")
print(feature_importance_df.head(5))
