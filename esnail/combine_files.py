import pandas as pd

df_starry = pd.read_csv("x2yz_compounds.csv")
df_mine = pd.read_csv("Full Heusler(X2YZ).csv")

# Make units match
df_starry['Seebeck Coefficient'] = df_starry['Seebeck Coefficient'] * 1000000

# Fill missing resistivity values with 1/conductivity
df_mine['Electrical Resistivity'] = df_mine['Electrical Resistivity'].fillna(1 / df_mine['Electrical Conductivity'])

# Verify which columns will be dropped
starry_only = set(df_starry.columns) - set(df_mine.columns)
mine_only = set(df_mine.columns) - set(df_starry.columns)

print(f"Columns only in Starrydata (will be dropped): {starry_only}")
print(f"Columns only in Extracted Data (will be dropped): {mine_only}")

# Merge both datasets
df_combined = pd.concat([df_starry, df_mine], join='inner', ignore_index=True)

output_file = "combined_full_heusler_dataset.csv"
df_combined.to_csv(output_file, index=False)

print(f"\nSuccessfully combined! Final shape: {df_combined.shape}")
print(f"Saved to: {output_file}")