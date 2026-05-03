import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

import re
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

def fix_fractional_formulas(formula_str):
    if pd.isna(formula_str):
        return formula_str
        
    formula_str = str(formula_str)
    
    # This Regex pattern looks for: (Element1 num/den Element2 num/den)Multiplier
    pattern = r'\(([A-Z][a-z]?)(\d+)/(\d+)([A-Z][a-z]?)(\d+)/(\d+)\)(\d+\.?\d*)'
    
    match = re.search(pattern, formula_str)
    if match:
        el1, n1, d1, el2, n2, d2, multiplier = match.groups()
        
        # Calculate the actual distributed amounts
        val1 = (float(n1) / float(d1)) * float(multiplier)
        val2 = (float(n2) / float(d2)) * float(multiplier)
        
        replacement = f"{el1}{val1:.3f}{el2}{val2:.3f}"
        
        fixed_formula = re.sub(pattern, replacement, formula_str)
        return fixed_formula
        
    return formula_str

if __name__ == "__main__":
    df = pd.read_csv("Full Heusler(X2YZ).csv")
    df = df.drop(columns=["Crystal Category", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16", "Unnamed: 17", "Unnamed: 18", "Unnamed: 19", "Paper (hyperlink)"])

    df = df.dropna(subset=["Formula"]).reset_index(drop=True)
    # bad_formulas = df[df['Formula'].str.contains('Val', na=False)]['Formula'].tolist()
    # if bad_formulas:
    #     print(f"Found and fixing typos in: {bad_formulas}")
        
    # # Replace the lowercase 'a' with a capital 'A'
    # df['Formula'] = df['Formula'].str.replace('Val', 'VAl')
    df['Formula'] = df['Formula'].apply(fix_fractional_formulas)
    
    
    print("Starting featurization")
    print("Parsing formulas...")
    df = StrToComposition(target_col_id='composition').featurize_dataframe(df, "Formula")

    print("Calculating physical properties (this may take a moment)...")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")

    df = ep_feat.featurize_dataframe(df, col_id='composition', ignore_errors=True)

    print("Done! Here are your new Machine Learning features:")

    new_cols = [col for col in df.columns if "MagpieData" in col]
    print(new_cols[:10])
    print(df.columns)

