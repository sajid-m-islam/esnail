import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def reduce_features(df, output): 
    # Handle any text values in the seebeck and resistivity columns
    df["Seebeck Coefficient"] = pd.to_numeric(df["Seebeck Coefficient"], errors="coerce")
    df["Electrical Resistivity"] = pd.to_numeric(df["Electrical Resistivity"], errors="coerce")

    # Drop any columns with nan
    df = df.dropna(subset=["Seebeck Coefficient", "Electrical Resistivity"]).reset_index(drop=True)

    # Assign target and features, drop all non-numerical data
    target = "Seebeck Coefficient"
    cols_to_drop = ["Composition", "Site_X", "Site_Y", "Site_Z", "Seebeck Coefficient", "Electrical Resistivity"]

    X = df.drop(columns=cols_to_drop)
    y = df[target]

    # Use imputer to fill missing values
    imputer = SimpleImputer(strategy="median")
    X_final = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Rank feature importance
    print(f"Training Random Forest to predict {target}...")
    print(f"Number of starting features: {X_final.shape[1]}")

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_final, y)

    importances = rf.feature_importances_

    # Create data frame ranking the most important features
    feature_importance_df = pd.DataFrame({ 
        "Feature": X.columns, 
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Display top 20 features
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df["Feature"][:20][::-1], feature_importance_df["Importance"][:20][::-1])
    plt.xlabel("RandomForest Importance Score")
    plt.title(f"Top 20 Most Important Features for {target}")
    plt.tight_layout()
    plt.show()

    # Get top 40 features
    top_features = feature_importance_df["Feature"][:40].tolist()

    # Create new dataframe with reduced features and save in csv
    final_cols = ["Composition", "Site_X", "Site_Y", "Site_Z"] + top_features + ["Seebeck Coefficient", "Electrical Resistivity"]

    df_reduced = df[final_cols]
    df_reduced.to_csv(output, index=False)

    print("\n" + "="*40)
    print(f"Dimensionality Reduction Complete")
    print(f"Reduced from {X.shape[1]} features down to 40.")
    print("="*40)
        
    print("\nTop 5 absolute best predictors:")
    print(feature_importance_df.head(5))

if __name__ == "__main__":
    df = pd.read_csv("./data/heusler_expanded_variance.csv")
    reduce_features(df, "./data/heusler_reduced_features_NEW.csv")
