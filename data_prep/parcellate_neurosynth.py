# ------------------------------------------------------------------------------------------------ 
# --- Parcellate Neurosynth annotations --- #
# ------------------------------------------------------------------------------------------------ 

# This notebook is adapted from [Justine Hansen's tutorial]('https://github.com/netneurolab/ipn-summer-school/blob/main/lectures/2021-07-02/13-15/fetch_and_parcellate_neurosynth.ipynb') for fetching and parcellating Neurosynth terms. *Note that Neurosynth's package is now deprecated, so all of this code has been refactored using NiMARE!*

# The script does the following: 
# 1. Fetch Neurosynth meta-analyses (association tests) for all available Cognitive Atlas concepts.
#     + Download raw Neurosynth data terms
#     + Convert Neurosynth database to NiMARE dataset file
#     + Run NiMARE's Multilevel kernel density analysis (MKDA) - Chi-square analysis to generate association z-value maps (i.e., reverse inference)
# 2. Concepts are parcellated according to the 360-node Glasser (HCP-MMP) parcellation.


# ------------------------------------------------------------------------------------------------ 
# --- Import packages --- #
# ------------------------------------------------------------------------------------------------ 

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
import os
import pandas as pd
import requests
import gzip
import pickle
import json
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2
import sys
import yaml
sys.path.append('/Users/joelleba/PennLINC/tractmaps/code')
from data_prep import tm_parcellate
from utils import tm_utils

# ------------------------------------------------------------------------------------------------ 
# --- Define inputs --- #
# ------------------------------------------------------------------------------------------------ 

# root
root = '/Users/joelleba/PennLINC/tractmaps'

# this is where the raw and parcellated data will be stored
ns_dir = '/Users/joelleba/Documents/large_data/tractmaps/neurosynth/raw/'
parc_dir = f'{root}/data/derivatives/neurosynth_annotations/'


# we'll use association z-value maps generated using neurosynth-style NiMARE meta-analyses (these were also called 'specificity' in older NiMARE versions)
# can add 'z_desc-uniformity_level-voxel_corr-FDR_method-indep' plus more, if desired
# ma_images = ['z_desc-association_level-voxel_corr-FDR_method-indep'] # if you want FDR corrected maps (not used in this analysis, see below)
ma_images = ['z_desc-association']

# Load configuration for email address
config_path = f'{root}/code/data_prep/parcellate_neurosynth_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# define email address (needed to download abstracts from neurosynth)
email_address = config['email_address']

# load df with tracts probs and glasser labels 
tracts_regs_ids = pd.read_csv(f'{root}/data/derivatives/tracts/tracts_probabilities/tracts_probabilities.csv')


# ------------------------------------------------------------------------------------------------ 
# --- Create output folder --- #
# ------------------------------------------------------------------------------------------------ 
# create output directories if they don't yet exist
if not os.path.exists(ns_dir):
    os.makedirs(ns_dir, exist_ok=True)
    print(f"Folder '{ns_dir}' created.")
else:
    print(f"Folder '{ns_dir}' already exists.")

if not os.path.exists(f'{parc_dir}glasser'):
    os.makedirs(f'{parc_dir}glasser', exist_ok=True)
    print(f"Folder '{parc_dir}glasser' created.")
else:
    print(f"Folder '{parc_dir}glasser' already exists.")

# ------------------------------------------------------------------------------------------------ 
# --- Define functions --- #
# ------------------------------------------------------------------------------------------------ 
def get_cogatlas_concepts(url = None):
    """ 
    Fetches list of concepts from the Cognitive Atlas.
    
    Parameters
    ----------
    url : str
        URL to Cognitive Atlas API
    Returns
    -------
    concepts : set
        Unordered set of terms
    """

    if url is None:
        url = 'https://cognitiveatlas.org/api/v-alpha/concept'

    req = requests.get(url)
    req.raise_for_status()
    concepts = set([f.get('name') for f in json.loads(req.content)])

    return concepts

def run_meta_analyses(database, ma_images, use_features = None, outdir = None):
    """
    Runs NiMARE-style meta-analysis based on a NiMARE `Dataset` object.
    
    Parameters
    ----------
    database : NiMARE object of class `Dataset`.
        Storage container for a coordinate- and/or image-based meta-analytic dataset/database.
        Note: NiMARE database should be generated prior to running this function.
    use_features : list, optional
        List of features on which to run NS meta-analyses; if not supplied all
        terms in `features` will be used
    outdir : str or os.PathLike
        Path to output directory where derived files should be saved
    Returns
    -------
    generated : list of str
        List of filepaths to generated term meta-analysis directories
    """

    # check outdir
    if outdir is None:
        outdir = os.path.join(ns_dir, 'terms')
    
    # extract list of labels for which studies in Dataset have annotations
    all_features = set(database.get_labels()) 
    features = [f.replace("terms_abstract_tfidf__", "") for f in all_features] 
    
    # if we only want a subset of the features take the set intersection
    if use_features is not None:
        features = set(features) & set(use_features)
       
    # empty list to store meta-analytic outputs
    generated = [] 
    
    # Progress tracking
    total_terms = len(sorted(features))
    print(f'Running meta-analyses for {total_terms} terms...')
    
    for i, word in enumerate(sorted(features), 1):
        # Progress bar
        progress = (i / total_terms) * 100
        bar_length = 30
        filled_length = int(bar_length * i // total_terms)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Clean message with progress bar
        msg = f'[{bar}] {i}/{total_terms} ({progress:.1f}%) - Processing: {word}'
        print(msg, end='\r', flush=True)

        # run meta-analysis and save specified outputs (only if they don't exist)
        path = os.path.join(outdir, word.replace(' ', '_'))
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        # Check if all meta-analysis output nii.gz images already exist in the term's directory
        # Only run meta-analysis if any output files are missing
        if not all(os.path.exists(os.path.join(path, f'{f}.nii.gz')) for f in ma_images):
            
            # get list of ids for each term of interest 
            ids = database.get_studies_by_label(f'terms_abstract_tfidf__{word}', 0.001)

            # Select studies with term of interest
            dset_sel = database.slice(ids) 

            # Select studies without term of interest
            no_term_ids = list(set(database.ids).difference(ids))
            dset_unsel = database.slice(no_term_ids)
            
            # Do Multilevel kernel density analysis (MKDA) - Chi-square analysis for coordinate-based meta-analysis
            # (NiMARE's estimator that matches Neurosynth's MetaAanalysis class)
            mkda = MKDAChi2(kernel__r = 10) # kernel radius of 10
            results = mkda.fit(dset_sel, dset_unsel) # fit MDKAChi2 model to studies with term of interest vs rest
            
            # FDR correction (voxel-level correction, i.e. the voxelwise z-statistics are adjusted)
#             corr = FDRCorrector(method = "indep", alpha = 0.05)
#             cres = corr.transform(results)
            
            # save out output maps specified in the ma_images list
            results.save_maps(path, names = ma_images)
#             cres.save_maps(path, names = ma_images)

        # store path to meta-analysis for the word
        generated.append(path)

    # Clear the progress bar and print completion message
    print(' ' * len(msg), end='\r', flush=True)
    print(f'✓ Completed meta-analyses for {total_terms} terms!')

    return generated

# -------------------------------------------------------------------------------------------------------
# --- Fetch Neurosynth meta-analyses (association tests) for all available Cognitive Atlas concepts --- #
# -------------------------------------------------------------------------------------------------------
# Neurosynth’s data files are stored at https://github.com/neurosynth/neurosynth-data.
print('Fetching Neurosynth meta-analyses...')

# get neurosynth coordinates, metadata and vocabulary terms
files = fetch_neurosynth(
    data_dir = ns_dir,
    version = "7",
    overwrite = False,
    source = "abstract",
    vocab = "terms",
)
neurosynth_db = files[0] 

# path to neurosynth dataset
output_file = os.path.join(ns_dir, 'neurosynth/neurosynth_dataset.pkl.gz')

# if absent, convert Neurosynth/NeuroQuery database files into NiMARE Dataset (take a little while)
print('Converting Neurosynth database to NiMARE Dataset...')
if not os.path.exists(output_file):
    neurosynth_dset = convert_neurosynth_to_dataset(
        coordinates_file = neurosynth_db["coordinates"],
        metadata_file = neurosynth_db["metadata"],
        annotations_files = neurosynth_db["features"],
    )
    neurosynth_dset.save(output_file)
    print(neurosynth_dset)

#     neurosynth_dset = download_abstracts(neurosynth_dset, email_address) # if you want to download abstracts; not used in this analysis
#     neurosynth_dset.save(os.path.join(ns_dir, "neurosynth/neurosynth_dataset_with_abstracts.pkl.gz"))
else:
    print(f"{output_file} already exists; skipping this step.")


# run meta-analyses 
print('Running meta-analyses...')
# with gzip.open(os.path.join(ns_dir, "neurosynth/neurosynth_dataset_with_abstracts.pkl.gz"), 'rb') as f:
with gzip.open(os.path.join(ns_dir, "neurosynth/neurosynth_dataset.pkl.gz"), 'rb') as f:
    neurosynth_dset = pickle.load(f)
    
# run NS meta-analyses on concepts from Cognitive Atlas (907 terms)
# when intersected with available terms from NeuroSynth (done in fct below), the total number of terms we'll analyze is N = 124
generated = run_meta_analyses(database = neurosynth_dset, 
                              ma_images = ma_images,
                              use_features = get_cogatlas_concepts()
                             )
print('Done with meta-analyses!')

# -------------------------------------------------------------------------------------------------------
# --- Parcellate Neurosynth annotations --- #
# -------------------------------------------------------------------------------------------------------

print('Parcellating neurosynth annotations...')

# Directory containing all the folders
root_dir = f'{ns_dir}/terms'

# Create the cogatl_maps dictionary using dictionary comprehension
cogatl_maps = {
    term_name: {
        'annotation': ('cognitive atlas', term_name, 'MNI152'), # space (last term) will be used in glasserize() to parcellate data
        'map': os.path.join(root_dir, term_name, f'z_desc-association.nii.gz')
    }
    for term_name in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, term_name))
}

print(f'Loaded {len(cogatl_maps)} cognitive terms into the dictionary.')

# generate glasser parcellated neurosynth maps 
glasser_cogatl_maps = tm_parcellate.glasserize(cortical_maps = cogatl_maps, zscore = False) # z-scoring not applied as NS maps are already z-scored
print(f'Applied Glasser parcellation!')

# -------------------------------------------------------------------------------------------------------
# --- Save parcellated neurosynth annotations --- #
# -------------------------------------------------------------------------------------------------------

# count number of terms
n_terms = len(glasser_cogatl_maps)

# # save cognitive atlas terms dictionary for analyses
# tm_utils.save_data(glasser_cogatl_maps, f'{parc_dir}glasser/glasser_neurosynth_{n_terms}terms_dict.pickle')

# Transform cortical maps into dataframe
df_glasser_cogatl_maps = pd.DataFrame(glasser_cogatl_maps)
df_glasser_cogatl_maps['regionID'] = range(1, len(df_glasser_cogatl_maps) + 1)

# a little bit of cleaning
sorted_columns = sorted(df_glasser_cogatl_maps.columns.difference(['regionID'])) # sort the cognitive terms columns alphabetically
df_glasser_cogatl_maps = pd.concat([df_glasser_cogatl_maps['regionID'], df_glasser_cogatl_maps[sorted_columns]], axis = 1) # reorder columns

# Merge tracts probability maps and cortical maps (rows are Glasser regions)
df_glasser_cogatl_maps_tracts = pd.merge(tracts_regs_ids, df_glasser_cogatl_maps, on = 'regionID')

# save to csv
df_glasser_cogatl_maps_tracts.to_csv(f'{parc_dir}glasser/glasser_tracts_neurosynth_{n_terms}terms.csv', header = True, index = False)
print(f'Saved {n_terms} terms!')
print(f'Saved to: {parc_dir}glasser/glasser_tracts_neurosynth_{n_terms}terms.csv')