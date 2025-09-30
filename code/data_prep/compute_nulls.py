"""
Generate indices nulls

This script generates spatial nulls for brain region indices using the 
Alexander-Bloch spin test approach. The nulls are used for statistical 
testing in PLS analyses.

Usage:
    python compute_nulls.py
"""

import os
import sys
from pathlib import Path
from neuromaps import nulls
sys.path.append(str(Path(__file__).parent.parent))
from utils import tm_utils


def generate_indices_nulls(root='/Users/joelleba/PennLINC/tractmaps/data', nspins=10000):
    """
    Generate nulls for indices (brain region indices only)
    
    Parameters:
    -----------
    root : str
        Root data directory
    nspins : int
        Number of spins to generate (default: 10000)
    
    Returns:
    --------
    spins : numpy.ndarray
        Array of shape (360, nspins) containing the resampling indices
    """
    nulls_dir = f'{root}/derivatives/nulls/'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(nulls_dir):
        os.makedirs(nulls_dir)
        print(f"Folder '{nulls_dir}' created.")
    else:
        print(f"Folder '{nulls_dir}' already exists.")
    
    # Load Glasser parcellation
    glasser = [os.path.join(root, 'derivatives/glasser_parcellation/HCP_MMP_L.label.gii'), # left hemisphere 
               os.path.join(root, 'derivatives/glasser_parcellation/HCP_MMP_R.label.gii')] # right hemisphere
    
    # Generate spins (no data provided so it returns resampling array)
    print(f"Generating {nspins} spatial nulls...")
    spins = nulls.alexander_bloch(data=None, atlas='fsLR', density='32k', 
                                  n_perm=nspins, seed=1234, parcellation=glasser)
    
    # Save nulls
    output_file = f'{nulls_dir}/indices_{nspins}spins.pickle'
    tm_utils.save_data(spins, output_file)
    print(f"Indices nulls saved to: {output_file}")
    print(f"Shape: {spins.shape}")
    
    return spins


if __name__ == "__main__":
    # Generate the indices nulls
    spins = generate_indices_nulls()
    print(f"Shape: {spins.shape}")
    print("Null generation complete!")

