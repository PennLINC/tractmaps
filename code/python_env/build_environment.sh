# ------------------------------------------------------------------------------------------------
# --- Build Python environment ---
# ------------------------------------------------------------------------------------------------

conda create -n tractmaps python=3.8
conda activate tractmaps

# Essentials
pip install pandas numpy seaborn matplotlib nibabel nilearn openpyxl statsmodels mapalign wordcloud scipy

# Neuromaps (make sure Connectome Workbench is installed prior to this - download at https://www.humanconnectome.org/software/get-connectome-workbench, 
# and don't forget to add the path to your bash profile:  echo 'export PATH=$PATH:/Applications/workbench/bin_macosx64' >> ~/.bash_profile)
# note that for newer Macs, you need to add the path like so echo 'export PATH=$PATH:/Applications/workbench/bin_macosxsub' >> ~/.zshrc

# Make software directory if it doesn't exist
software_dir='/Users/joelleba/PennLINC/tractmaps/software'
mkdir -p "$software_dir"

# neuromaps
cd $software_dir
git clone https://github.com/netneurolab/neuromaps.git
cd neuromaps
pip install .

# abagen
pip install abagen

# brainsmash
pip install brainsmash

# PLS
pip install cython # required for new mac otherwise pyls doesn't install properly
brew install pkg-config hdf5  # required for new mac otherwise pyls doesn't install properly
HDF5_DIR=$(brew --prefix hdf5) pip install h5py
cd $software_dir
git clone https://github.com/rmarkello/pyls.git
cd pyls
python setup.py install

# Netneurotools and dependencies
conda install PyQt5 
conda install vtk 
conda install mayavi
pip install netneurotools 

# install hcp utils (useful for Glasser parcellation) and dependencies
pip install hcp_utils

# install nimare (to fetch Neurosynth data)
pip install nimare markupsafe biopython

# save environment requirements in the script's directory
script_dir="$(dirname "$0")"
conda env export > "$script_dir/environment.yml" 