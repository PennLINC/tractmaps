######################################################################################################################
#### PARCELLATE FUNCTIONS ####
######################################################################################################################

# This script contains functions to load neuromaps annotations and parcellate them using the Glasser parcellation.

######################################################################################################################

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from neuromaps.datasets import fetch_annotation
import hcp_utils as hcp
from neuromaps import images
from neuromaps.parcellate import Parcellater
from scipy import stats as sstats

### get Glasser parcellation ###
def get_glasser(lh_glasser = None, rh_glasser = None):
    """
    Create GIFTI objects for the left and right hemispheres using the Glasser atlas labels and data. GIFTI objects are outputted as a tuple. 
    
    Parameters:
    -----------
        lh_glasser (optional): GIFTI object for the left hemisphere. If provided, custom data will be used for the left hemisphere.
        rh_glasser (optional): GIFTI object for the right hemisphere. If provided, custom data will be used for the right hemisphere.
    
    Returns:
    --------
        tuple: A tuple containing GIFTI objects for the left and right hemispheres.

    If custom GIFTI objects are not provided, the function generates GIFTI objects based on the Glasser atlas labels and data.

    Example Usage:
        glasser = get_glasser()
    """
    
    # select region labels
    labels = list(hcp.mmp.labels.values())

    # redefine the medial wall label to match one of the elements in PARCIGNORE 
    # (required for generating spin samples in the alexander_bloch nulls function)
    labels[0] = 'medial_wall'

    # select left and right hemisphere labels
    lh_labels = labels[0:181] # the first is unassigned (0) and 1-180 are lh labels
    rh_labels = [labels[0]]  + labels[181:361] # the first (0) unassigned label and 181-360 are rh labels
    
    # if left and right data are not provided, use the hcp glasser data
    if lh_glasser is None:
        # Transform array of glasser labels in grayordinates (hcp.mmp.map_all, size=59412) to labels in vertices (size=32492) 
        # The unused vertices are filled with a constant (zero by default). Order is Left, Right hemisphere.
        lh_glasser_verts = hcp.left_cortex_data(hcp.mmp.map_all)
        
        # create gifti
        lh_glasser = images.construct_shape_gii(lh_glasser_verts, labels = lh_labels,
                                           intent = 'NIFTI_INTENT_LABEL')
        
    if rh_glasser is None:
        rh_glasser_verts = hcp.right_cortex_data(hcp.mmp.map_all)
        rh_glasser = images.construct_shape_gii(rh_glasser_verts, labels = rh_labels,
                                       intent = 'NIFTI_INTENT_LABEL')
    
    # create tuple of left and right hemisphere GIFTI images
    glasser = (lh_glasser, rh_glasser)
    
    return glasser

### Apply Glasser parcellation to cortical maps ###
def glasserize(cortical_maps, zscore = True):
    """
    Parcellate cortical maps using the Glasser parcellation and store the parcellated maps in a dictionary.

    Parameters:
    -----------
        cortical_maps (dict): A dictionary of cortical maps to be parcellated.
        zscore (bool, optional): Whether to z-score the parcellated cortical maps. Defaults to True.

    Returns:
    --------
        dict: A dictionary containing the parcellated maps.
    """
    
    # get glasser object (a tuple containing Glasser labels as GIFTI objects for the left and right hemispheres)
    glasser = get_glasser()
    
    # create parcellater object
    glasser_parc = Parcellater(parcellation = glasser,
                               space = 'fsLR', # space in which the parcellation is defined
                               resampling_target = 'parcellation') # cortical maps provided later will be resampled to the space + res of the parcellation
    # glasser_parc.parcellation[0].darrays[0] #.data # to look at the data

    # parcellate input maps
    glasser_maps = {}  
    for map_name, value in cortical_maps.items():
        print(f'Map: {map_name}')

        # apply Glasser parcellation to the map
        parcellated_map = glasser_parc.fit_transform(data = value['map'], space = value['annotation'][2])

        # z-score the map if zscore is True
        if zscore:
            parcellated_map = sstats.zscore(parcellated_map, nan_policy = 'omit')

        glasser_maps[f'{map_name}'] = parcellated_map

        # Load non-parcellated maps to compare data shape with parcellated maps
        data = images.load_data(value['map'])
        print(f'Original map shape: {data.shape}, parcellated shape: {parcellated_map.shape}') 

    return glasser_maps
