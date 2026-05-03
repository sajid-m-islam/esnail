import pandas as pd
try:
    from pymatgen.core import Composition
except ImportError:
    print("Please install pymatgen first by running: pip install pymatgen pandas")
    exit(1)

def is_x2yz(formula_str, tolerance=0.04):
    """
    Checks if a chemical formula string matches the X2YZ stoichiometry structure.
    Allows for doping (up to 6 elements) by grouping elements into 3 sublattices
    and checking if their combined atomic fractions equal ~50%, ~25%, and ~25%.
    """
    try:
        if pd.isna(formula_str):
            return False
            
        comp = Composition(str(formula_str))
        
        reduced_comp, _ = comp.get_reduced_composition_and_factor()
        reduced_atoms = reduced_comp.num_atoms
        
        if not (3.5 <= reduced_atoms <= 5.5):
            return False
            
        elements = comp.elements

        
        # Rule 1: Allow between 3 to 6 elements to account for standard doping,
        # but prevent massive clathrates or high-entropy alloys.
        if not (3 <= len(elements) <= 6):
            return False
            
        # Rule 2: Exclude purely non-transition metal alloys
        has_tm_or_re = any(el.is_transition_metal or el.is_lanthanoid or el.is_actinoid for el in elements)
        if not has_tm_or_re:
            return False
            
        # Rule 3: Exclude oxides, nitrides, halides, and chalcogenides
        excluded_elements = ['O', 'F', 'Cl', 'Br', 'I', 'S', 'N', 'Te', 'Se', 'C', 'H']
        if any(el.symbol in excluded_elements for el in elements):
            return False
            
        # Rule 4: Group elements into 3 sublattices (X, Y, Z) and check 50/25/25 fractions
        fractions = [comp.get_atomic_fraction(el) for el in elements]
        n = len(fractions)
        
        # Test all possible ways to assign 'n' elements into 3 bins (representing the 3 sites)
        for i in range(3**n):
            bins = [0.0, 0.0, 0.0]
            bin_elements = [[], [], []] # Track which elements go into which site
            temp = i
            for j in range(n):
                site_idx = temp % 3
                bins[site_idx] += fractions[j]
                bin_elements[site_idx].append(elements[j])
                temp //= 3
                
            # All 3 sublattices must be occupied (no empty sites)
            if any(b == 0.0 for b in bins):
                continue
                
            sorted_bins = sorted(bins)
            expected_fractions = [0.25, 0.25, 0.50]
            
            # Check if this specific grouping matches the target fractions within our tolerance
            match = True
            for actual, expected in zip(sorted_bins, expected_fractions):
                if abs(actual - expected) > tolerance:
                    match = False
                    break
                    
            if match:
                # --- NEW RULE: Enforce the X2 site ---
                fifty_percent_bin = -1
                for b_idx, b_val in enumerate(bins):
                    if abs(b_val - 0.50) <= tolerance:
                        fifty_percent_bin = b_idx
                        break
                        
                if fifty_percent_bin != -1:
                    x_site_elements = bin_elements[fifty_percent_bin]
                    # Check if AT LEAST ONE of the elements on the 50% site is a TM/RE.
                    is_valid_x_site = any(el.is_transition_metal or el.is_lanthanoid or el.is_actinoid for el in x_site_elements)
                    
                    if is_valid_x_site:
                        return True
                
        return False
        
    except Exception:
        return False

def main():
    input_file = 'interpolated_data.csv'
    output_file = 'x2yz_compounds.csv'
    
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_file}' in the current directory.")
        return

    col_name = 'composition'
    if col_name not in df.columns:
        print(f"Error: '{col_name}' column not found.")
        return

    print("Parsing compositions and filtering for structurally matched X2YZ compounds (including doped)...")
    mask = df[col_name].apply(lambda x: is_x2yz(x, tolerance=0.04))
    
    # Extract matching rows
    x2yz_df = df[mask].reset_index(drop=True)
    
    # --- FIX: Drop rows that are missing ML target data ---
    target_cols = ['Seebeck coefficient', 'Electrical resistivity']
    
    if all(col in x2yz_df.columns for col in target_cols):
        # Forces both properties to be present (non-NaN).
        x2yz_df = x2yz_df.dropna(subset=target_cols, how='any').reset_index(drop=True)
    else:
        print(f"Warning: One or both of the exact columns {target_cols} were not found in the dataset.")
    
    # Save to CSV
    x2yz_df.to_csv(output_file, index=False)
    
    print("-" * 40)
    print(f"Filtering complete!")
    print(f"Original dataset rows: {len(df)}")
    print(f"Found structural X2YZ compounds: {len(x2yz_df)}")
    print(f"Saved to: {output_file}")
    
    # --- NEW: ML Data Summary Section ---
    if not x2yz_df.empty:
        print("\n" + "="*40)
        print("UNIQUE COMPOUNDS FOUND:")
        print("="*40)
        unique_compounds = x2yz_df[col_name].unique()
        print(f"Total Unique Compositions: {len(unique_compounds)}")
        print(unique_compounds[:10]) 

        print("\n" + "="*40)
        print("MACHINE LEARNING FEATURES SUMMARY:")
        print("="*40)
        
        # Look strictly for exact ML column names
        ml_cols = ['composition', 'Temperature', 'Seebeck coefficient', 'Electrical resistivity']
        existing_ml_cols = [col for col in ml_cols if col in df.columns]
        
        # Show a preview of just the ML-relevant columns
        print("\nData Preview:")
        print(x2yz_df[existing_ml_cols].head())
        
        print("\nUsable Data Points (Non-Null values):")
        for col in existing_ml_cols:
            valid_count = x2yz_df[col].notna().sum()
            print(f"- {col}: {valid_count} valid rows")

if __name__ == "__main__":
    main()