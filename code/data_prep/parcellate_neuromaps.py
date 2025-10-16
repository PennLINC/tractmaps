# ------------------------------------------------------------------------------------------------ #
# --- Parcellate Neuromaps annotations --- #
# ------------------------------------------------------------------------------------------------ #

# import packages
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from data_prep import tm_parcellate
from utils import tm_utils
from neuromaps.datasets import fetch_annotation

# ------------------------------------------------------------------------------------------------ 
# --- Load data --- #
# define root
root = '/Users/joelleba/PennLINC/tractmaps/data'

# load tract-to-region mapping
tracts_regs_ids = pd.read_csv(f'{root}/derivatives/tracts/tracts_probabilities/tracts_probabilities.csv')


# ------------------------------------------------------------------------------------------------ 
# --- Create output folder --- #
# create folder to store inputs for data analyses if doesn't yet exist
output_folder = os.path.join(root, 'derivatives/cortical_annotations/glasser/')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' created.")
else:
    print(f"Folder '{output_folder}' already exists.")


# ------------------------------------------------------------------------------------------------ 
# --- Fetch Neuromaps annotations --- #
# keys are shorter map names; values should be a list of tuples with 4 elements each
nm_annotations_dict = {
                    'myelin': ('hcps1200', 'myelinmap', 'fsLR', '32k'),
                    'genes_pc1': ('abagen', 'genepc1', 'fsaverage', '10k'),
                    'pet_gaba': ('norgaard2021', 'flumazenil', 'fsaverage', '164k'),
                    'pet_h5t1b': ('beliveau2017', 'az10419369', 'fsaverage', '164k'),
                    'pet_5ht2a': ('beliveau2017', 'cimbi36', 'fsaverage', '164k'),
                    'pet_5ht4': ('beliveau2017', 'sb207145', 'fsaverage', '164k'),
                    'pet_5ht1a': ('savli2012', 'way100635', 'MNI152', '3mm'),
                    'pet_htt': ('beliveau2017', 'dasb', 'fsaverage', '164k'),
                    'pet_ht6': ('radnakrishnan2018', 'gsk215083', 'MNI152', '1mm'),
                    'pet_d1': ('kaller2017', 'sch23390', 'MNI152', '3mm'),
                    'pet_d2': ('sandiego2015', 'flb457', 'MNI152', '1mm'),
                    'pet_achm1': ('naganawa2020', 'lsn3172176', 'MNI152', '1mm'),
                    'pet_mglur5': ('smart2019', 'abp688', 'MNI152', '1mm'),
                    'pet_mor': ('kantonen2020', 'carfentanil', 'MNI152', '3mm'),
                    'pet_net': ('ding2010', 'mrb', 'MNI152', '1mm'),
                    'pet_sv2a': ('finnema2016', 'ucbj', 'MNI152', '1mm'),
                    'pet_vacht': ('aghourian2017', 'feobv', 'MNI152', '1mm'),
                    'pet_ach': ('hillmer2016', 'flubatine', 'MNI152', '1mm'),
                    'pet_cannab': ('normandin2015', 'omar', 'MNI152', '1mm'),
                    'pet_hist': ('gallezot2017', 'gsk189254', 'MNI152', '1mm'),
                    'pet_dat': ('dukart2018', 'fpcit', 'MNI152', '3mm')
                  }

# fetch neuro annotations
print(f'Fetching Neuromaps annotations...')
maps_dict = {}
for map_name, annotation in nm_annotations_dict.items():
    
    # print(f'Map: {map_name}, Annotation: {annotation}')
    fetched_map = fetch_annotation(source = annotation[0], desc = annotation[1], space = annotation[2], den = annotation[3])
    maps_dict[map_name] = {'annotation': annotation, 'map': fetched_map}

# add layer thickness maps from BigBrain
for i in range(6): 
    lh_file = (f'{root}/raw/tpl-fsaverage/tpl-fsaverage_hemi-L_den-164k_desc-layer{i+1}_thickness.shape.gii')
    rh_file = (f'{root}/raw/tpl-fsaverage/tpl-fsaverage_hemi-R_den-164k_desc-layer{i+1}_thickness.shape.gii')
    files = (lh_file, rh_file)
    map_name = f'layer{i+1}'
    maps_dict[map_name] = {'annotation': ('bigbrain', f'{map_name}', 'fsaverage', '164k'), 'map': files}
	
# show all maps and their annotations
for map_name, annotation in maps_dict.items():
    print(f'Map: {map_name}, Annotation: {annotation["annotation"]}')

# ------------------------------------------------------------------------------------------------ 
# --- Parcellate Neuromaps annotations --- #
print('Parcellating Neuromaps annotations...')
# generate glasser parcellated (and z-scored) cortical maps 
glasser_maps = tm_parcellate.glasserize(cortical_maps = maps_dict, zscore = True)


# ------------------------------------------------------------------------------------------------ 
# --- Save parcellated maps --- #
print('Saving parcellated maps...')
# # save dictionary for analyses
# tm_utils.save_data(glasser_maps, f'{output_folder}/glasser_tracts_corticalmaps_dict.pickle')

# transform cortical maps into dataframe
df_glasser_maps = pd.DataFrame(glasser_maps)
df_glasser_maps['regionID'] = range(1, len(df_glasser_maps) + 1)

# merge tracts probability maps and cortical maps (rows are Glasser regions)
df_glasser_maps_tracts = pd.merge(tracts_regs_ids, df_glasser_maps, on = 'regionID')

# save to csv
df_glasser_maps_tracts.to_csv(f'{output_folder}/glasser_tracts_corticalmaps.csv', index = False)

print(f'Number of maps: {len(maps_dict)}')
print(f'Saved to: {output_folder}/glasser_tracts_corticalmaps.csv')
print('Done!')
