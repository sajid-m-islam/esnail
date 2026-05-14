import pandas as pd
import numpy as np
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
import re
from pymatgen.core import Composition

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

def split_heusler_sites(formula_str):
    try:
        if pd.isna(formula_str):
            return pd.Series([np.nan, np.nan, np.nan])
            
        comp = Composition(formula_str)
        elements = comp.elements
        fractions = [comp.get_atomic_fraction(el) for el in elements]
        n = len(fractions)
        
        # Step A: Find the valid 50/25/25 grouping (tolerance allows for doping)
        best_bins = None
        tolerance = 0.04
        for i in range(3**n):
            bins = [0.0, 0.0, 0.0]
            bin_elements = [[], [], []]
            temp = i
            for j in range(n):
                site_idx = temp % 3
                bins[site_idx] += fractions[j]
                bin_elements[site_idx].append(elements[j])
                temp //= 3
                
            if any(b == 0.0 for b in bins):
                continue
                
            sorted_bins = sorted(bins)
            if (abs(sorted_bins[0] - 0.25) <= tolerance and 
                abs(sorted_bins[1] - 0.25) <= tolerance and 
                abs(sorted_bins[2] - 0.50) <= tolerance):
                best_bins = (bins, bin_elements)
                break
                
        if not best_bins:
            return pd.Series([np.nan, np.nan, np.nan])
            
        bins, bin_elements = best_bins
        
        # Step B: Identify the X site (the one holding ~50%)
        x_elements = []
        twenty_five_bins = []
        
        for b_val, b_elems in zip(bins, bin_elements):
            if abs(b_val - 0.50) <= tolerance and not x_elements:
                x_elements = b_elems
            else:
                twenty_five_bins.append(b_elems)
                
        if len(twenty_five_bins) != 2:
            return pd.Series([np.nan, np.nan, np.nan])
            
        # Step C: Differentiate Y and Z sites (~25% each)
        # Z site is the p-block/main group element (highest periodic table group number)
        def avg_group(elems):
            # Lanthanides/Actinides return group 3 in pymatgen, which naturally makes them the Y site!
            return sum(el.group for el in elems) / len(elems)
            
        if avg_group(twenty_five_bins[0]) < avg_group(twenty_five_bins[1]):
            y_elements = twenty_five_bins[0]
            z_elements = twenty_five_bins[1]
        else:
            y_elements = twenty_five_bins[1]
            z_elements = twenty_five_bins[0]
            
        # Step D: Reconstruct the site strings (normalized to exact atom counts)
        def create_site_string(elems, total_site_atoms):
            site_str = ""
            site_frac_sum = sum(comp.get_atomic_fraction(el) for el in elems)
            for el in elems:
                site_amt = (comp.get_atomic_fraction(el) / site_frac_sum) * total_site_atoms
                site_str += f"{el.symbol}{site_amt:.3f}"
            return site_str
            
        x_site_str = create_site_string(x_elements, 2.0)
        y_site_str = create_site_string(y_elements, 1.0)
        z_site_str = create_site_string(z_elements, 1.0)
        
        return pd.Series([x_site_str, y_site_str, z_site_str])
        
    except Exception:
        return pd.Series([np.nan, np.nan, np.nan])



if __name__ == "__main__":
    df = pd.read_csv("combined_full_heusler_dataset.csv")

    # Handle any fractional formulas
    df['Composition'] = df['Composition'].apply(fix_fractional_formulas)
    
    
    # Use function to split formulas into x, y, z-sites
    print("Mathematically splitting formulas into X, Y, and Z sites...")
    df[['Site_X', 'Site_Y', 'Site_Z']] = df['Composition'].apply(split_heusler_sites)
    
    # Drop rows where the splitting failed (e.g. non-Heuslers that slipped through)
    df = df.dropna(subset=['Site_X', 'Site_Y', 'Site_Z']).reset_index(drop=True)

    # Loop through each site and create matminer features for each site
    print("Starting site-specific featurization...")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    ep_feat.set_n_jobs(1) 
    
    for site in ['X', 'Y', 'Z']:
        print(f"\n--- Processing {site}-Site ---")
        
        df = StrToComposition(target_col_id=f'comp_{site}').featurize_dataframe(df, f'Site_{site}', ignore_errors=True)
        df = ep_feat.featurize_dataframe(df, col_id=f'comp_{site}', ignore_errors=True)

        # Rename and filter columns
        magpie_cols = [col for col in df.columns if "MagpieData" in col]
        
        # Drop the variance/disorder features for now
        dev_cols = [col for col in magpie_cols if "avg_dev" in col]
        df = df.drop(columns=dev_cols)
        
        # Re-fetch the list now that avg_dev is gone
        magpie_cols = [col for col in df.columns if "MagpieData" in col]
        
        rename_map = {
            "MagpieData mean ": f"{site}_Mean_",
            "MagpieData minimum ": f"{site}_Min_",
            "MagpieData maximum ": f"{site}_Max_",
            "MagpieData range ": f"{site}_Range_",
            "MagpieData mode ": f"{site}_Mode_"
        }
        
        new_names = {}
        for col in magpie_cols:
            new_col_name = col
            for old_text, new_text in rename_map.items():
                if old_text in new_col_name:
                    new_col_name = new_col_name.replace(old_text, new_text)
            new_names[col] = new_col_name
            
        df = df.rename(columns=new_names)
        df = df.drop(columns=[f'comp_{site}'])

    print("\n" + "="*40)
    print("Featurization Complete!")
    print("="*40)
    
    z_features = [col for col in df.columns if col.startswith("Z_Mean_")]
    print(df[['Composition', 'Site_X', 'Site_Y', 'Site_Z'] + z_features[:3]].head())
    
    # Save new dataset
    output_filename = "heusler_with_features_dataset.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nSaved {df.shape[1]} total columns to '{output_filename}'.")
    print(f"Shape of dataset: {df.shape}")

