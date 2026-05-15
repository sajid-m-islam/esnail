import pandas as pd
import numpy as np
from pymatgen.core import Composition
from matminer.utils.data import MagpieData

def calc_entropy(composition_str):
    if pd.isna(composition_str):
        return 0.0
    
    comp = Composition(composition_str)

    # Calculates the entropy using the formula : entropy = -sum(x_i * ln(x_i))
    # x_i is every element at a given site
    entropy = 0.0
    for element in comp.elements:
        fraction = comp.get_atomic_fraction(element)
        if fraction > 0:
            entropy -= fraction * np.log(fraction)

    return entropy

def calc_variance(composition_str, property, magpie_data):
    if pd.isna(composition_str):
        return 0.0
    
    comp = Composition(composition_str)

    # Sites with only one element have no variance
    if len(comp.elements) <= 1:
        return 0.0
    
    fractions = []
    property_values = []

    # Use the formula variance = sum(x_i * (P_i - Mean_P)^2)
    # x_i is all of the elements at a given site
    # P_i is a elemental property and Mean_P is a mean of that property 
    for element in comp.elements:
        try:
            # Look up the elemental property in the Magpie database
            val = magpie_data.get_elemental_property(element, property)
            if pd.isna(val):
                return 0.0
            property_values.append(val)
            fractions.append(comp.get_atomic_fraction(element))
        except:
            return 0.0
        
    # Calculate weighted mean    
    mean = sum(frac * prop for frac, prop in zip(fractions, property_values))

    # Calculate variance using mean
    variance = sum(frac * (prop - mean)**2 for frac, prop in zip(fractions, property_values))

    return variance

if __name__ == "__main__":
    df = pd.read_csv("./data/heusler_reduced_features.csv")

    magpie = MagpieData()

    # List of properties to calculate variance for
    properties_to_track = [
        "GSvolume_pa",      # Lattice strain / Size mismatch
        "GSbandgap",        # Electronic structure
        "GSmagmom",         # Magnetic moments 
        "AtomicWeight",     # Mass fluctuations 
        "MendeleevNumber",  # Chemical similarity sorting (Crucial for thermoelectrics)
        "Electronegativity",# Bond polarity / electron scattering
        "MeltingT",         # Thermodynamic stability / Bond strength
        "CovalentRadius",   # Atomic packing density
        "SpaceGroupNumber", # Symmetry breaking
        "NsUnfilled",       # s-band scattering / electron mobility
        "NpUnfilled",       # p-band scattering (crucial for Z-site main group)
        "NdUnfilled",       # d-band scattering (crucial for X/Y transition metals)
        "NfUnfilled",       # f-band scattering (crucial if rare-earth dopants are used!)
        "NsValence",        # s-orbital characteristics
        "NpValence",        # p-orbital characteristics
        "NdValence",        # d-orbital characteristics
        "NfValence",        # f-orbital characteristics 
        "NValence"          # Total valence electrons (overall charge fluctuations)
    ]

    sites = ["Site_X", "Site_Y", "Site_Z"]

    # Calculate the entropy and variance for all the sites and features
    for site in sites:
        df[f"{site}_entropy"] = df[site].apply(calc_entropy)

        for prop in properties_to_track:
            df[f"{site}_variance_{prop}"] = df[site].apply(lambda x: calc_variance(x, prop, magpie))
        
    df.to_csv("./data/heusler_expanded_variance.csv", index=False)

    print("\n" + "="*40)
    print("Phase 4 Complete: Entropy & Variance Features Added")
    print(f"Added 3 Entropy features and {3 * len(properties_to_track)} Variance features.")
    print(f"Final shape: {df.shape}")
    print("="*40)

    print("\nPreview of the newly generated features (First 3 rows, Z-Site sample):")
    new_cols = [col for col in df.columns if 'entropy' in col or 'variance' in col]
    
    z_sample_cols = [col for col in new_cols if 'Site_Z' in col][:5] 
    print(df[['Composition', 'Site_Z'] + z_sample_cols].head(3))

