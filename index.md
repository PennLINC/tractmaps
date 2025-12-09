---
layout: page
title: Reproducibility Guide
---

# Reproducibility guide

The entire analytic workflow implemented in this project is described in the following sections. This workflow includes data preparation and analyses included in the manuscript. Scripts should be run in the order outlined below.

# Table of contents

- [I. Project information](#i-project-information)
- [II. Directory structure](#ii-directory-structure)
- [III. Code documentation](#iii-code-documentation)
  - [Overview of the analytic workflow](#overview-of-the-analytic-workflow)
  - [Project software](#project-software)
    - [Connectome workbench](#connectome-workbench)
    - [DSI studio](#dsi-studio)
  - [Build the environment](#build-the-environment)
  - [Publicly available population-level data](#publicly-available-population-level-data)
  - [Data preparation](#data-preparation)
  - [Analysis](#analysis)
    - [Data structure (Figure 1)](#data-structure-figure-1)
    - [Spatial embedding (Figure 2)](#spatial-embedding-figure-2)
    - [Terms-tracts partial least squares (Figure 3)](#terms-tracts-partial-least-squares-figure-3)
    - [Functional decoding (Figure 4)](#functional-decoding-figure-4)
    - [Tract functional diversity (Figure 5)](#tract-functional-diversity-figure-5)
    - [Biological cortical similarity (Figure 6)](#biological-cortical-similarity-figure-6)
    - [Individual-level analysis (Figure 7)](#individual-level-analysis-figure-7)
  - [A note on utilities](#a-note-on-utilities)

# Project information

## Summary

While long-range white matter (WM) tracts support communication between distant regions, it remains unknown how WM tracts are positioned within the cortical hierarchy to support cognition. Here, we leveraged population-level tract-to-region mapping, task-based functional activations, and measures of cortical biology to delineate how tracts are situated within the cortical hierarchy to support cognition. We found that tracts are differentially positioned along the cortical hierarchy to support specific cognitive functions and serve as bridges connecting distinct biological environments. Their hierarchical placement also reflects developmental variation in tract microstructure and individual differences in cognition. Together, these findings provide a framework that moves beyond conventional tract categories and emphasizes the essential role that WM tracts play in the cortical hierarchy. 

## Team

| **Role** | **Name** |
| --- | --- |
| **Project lead** | Joëlle Bagautdinova |
| **Faculty lead** | Theodore D. Satterthwaite |
| **Analytic replicator** | Golia Shafiei |
| **Collaborators** | Audrey C. Luo,  Margaret K. Pecsok, Taylor Salo, Aaron F. Alexander-Bloch, Dani S. Bassett, Margaret E. Gardner, Raquel E. Gur, Ruben C. Gur, Allyson P. Mackey, Bratislav Misic, Tyler M. Moore, David R. Roalf, Russell T. Shinohara, Valerie J. Sydnor, Tien T. Tong, Fang-Cheng Yeh, Russell A. Poldrack, Matthew Cieslak |

## Timeline

| **Project start date** | January 2023 |
| --- | --- |
| **Current project status** | In prep |

## Code and communication

| **Github repository** | [https://github.com/PennLINC/tractmaps](https://github.com/PennLINC/tractmaps) |
| --- | --- |
| **Slack channel** | #cortical-tractometry |

## Datasets

- Population-level tract-to-region mappings from [Yeh, 2022](https://doi.org/10.1038/s41467-022-32595-4)
- Population-level meta-analytic task activations from [*Neurosynth*](https://neurosynth.org/)
- Population-level biological cortical properties from [*Neuromaps*](https://netneurolab.github.io/neuromaps/usage.html) and [*Bigbrain*](https://bigbrainwarp.readthedocs.io/en/latest/)
- Individual-level data from the [Philadelphia Neurodevelopmental Cohort](http://dx.doi.org/10.1016/j.neuroimage.2013.07.064) (PNC)

# Directory structure

Most data and analyses are done locally in the `tractmaps` directory. The directories on github contain everything but the `data/raw` folder. The directory is organized as follows: 

| **Directory** | **Description** |
| --- | --- |
| `~/code` | where manuscript code lives |
| `~/code/python_env` | has code to build the python environment |
| `~/code/data_prep` | raw data is used by code from `/code/data_prep` to generate derivatives used in the analyses |
| `~/code/get_data` | has code to get PNC data |
| `~/code/utils` | has utility functions used across analyses |
| `~/code/analysis` | has code to generate all manuscript result figures 1-7 |
| `~/data/` | where manuscript data lives |
| `~/data/raw` | has input data required to conduct analyses |
| `~/data/derivatives` | has data derivatives that are used in analyses |

Data for individual-level PNC analyses are stored on CUBIC. The project directory on CUBIC is `/cbica/projects/tractmaps`. The directory is organized as follows: 

| **Directory** | **Description** |
| --- | --- |
| `~/tractmaps/code/get_data` | code to fetch the PNC dMRI data and store in tractmaps cubic project |
| `~/data/PNC/` | has PNC dMRI scalar, qc, and demographics data for required in analyses |

# Code documentation

## Overview of the analytic workflow

| Step | Task |
| --- | --- |
| 1 | Install the required software and build the environment |
| 2 | Prepare data derivatives; these will be the inputs for analyses |
| 3 | Build the data structure figure (Figure 1) |
| 4 | Examine the relationship between tract spatial positioning and S-A range (Figure 2) |
| 5 | Assess the spatial relationship between tracts and cognitive terms using Partial Least Squares (Figure 3) |
| 6 | Perform functional decoding of WM tracts to get the contribution of each cognitive term in each tract (Figure 4) |
| 7 | Quantify the functional diversity of tracts using the Gini coefficient and evaluate the association with tract S-A range (Figure 5) |
| 8 | Quantify the biological cortical similarity of WM tracts and evaluate the relationship with S-A range and cognitive diversity (Figure 6) |
| 9 | Prepare PNC data for age and cognition analyses |
| 10 | Perform GAMs on PNC data assessing the effect of age and cognition on WM tracts microstructure (fractional anisotropy) |

## Project software

The following external software was used in this project:

- Connectome workbench, version 2.0.1 (see details below)
- DSI studio, version Hou "侯" May 27 2025 (see details below)
- Python, version 3.8
- R, version 4.4.2
- QSIPrep, version 1.0.0rc1
- QSIRecon, version 1.1.0

### Connectome workbench

Connectome Workbench can be downloaded at [https://www.humanconnectome.org/software/get-connectome-workbench](https://www.humanconnectome.org/software/get-connectome-workbench). This software is needed for generating region labels and for the neuromaps Python package. This project used version 2.0.1. 

Once that’s downloaded, add the path to the bash profile: 

```bash
echo 'export PATH=$PATH:/Applications/workbench/bin_macosx64' >> ~/.bash_profile)
# note that for newer Macs, add the path is added differently:
echo 'export PATH=$PATH:/Applications/workbench/bin_macosxsub' >> ~/.zshrc
```

### DSI studio

DSI studio can be downloaded at: [https://dsi-studio.labsolver.org/download.html](https://dsi-studio.labsolver.org/download.html)

This version was used for tract visualizations on a Mac book pro M4, MacOS 15.7:

```bash
DSI Studio version: Hou "侯" May 27 2025
```

## Build the environment

The Python environment used for this project was build using the build_environment.sh script.

Note: make sure `conda` and `homebrew` are installed prior to doing this!

```yaml
cd ~/code/python_env
bash build_environment.sh
```

## Publicly available population-level data

This project uses the following publicly available resources:

- The original HCP-MMP parcellation dlabel.nii from [https://balsa.wustl.edu/78X3](https://balsa.wustl.edu/78X3)
- Tract-to-region probability matrix from [Yeh 2022, Nature Methods](https://www.nature.com/articles/s41467-022-32595-4)
- Tract probability volumes from [https://github.com/frankyeh/data-atlas/releases/download/hcp1065/hcp1065_prob_coverage_nifti.zip](https://github.com/frankyeh/data-atlas/releases/download/hcp1065/hcp1065_prob_coverage_nifti.zip) listed on the dsi studio website: [https://brain.labsolver.org/hcp_trk_atlas.html](https://brain.labsolver.org/hcp_trk_atlas.html)
- Tract names and abbreviations : `data/derivatives/tract_names/abbreviations.xlsx` taken and adjusted from [https://github.com/data-others/atlas/releases/download/hcp1065/abbreviation2.xlsx](https://github.com/data-others/atlas/releases/download/hcp1065/abbreviation2.xlsx)
- S-A axis ranks, glasser parcellated from [https://github.com/PennLINC/S-A_ArchetypalAxis/tree/main/Glasser360_MMP](https://github.com/PennLINC/S-A_ArchetypalAxis/tree/main/Glasser360_MMP)

## Data preparation

First, raw data is used to prepare data derivatives that will be used in the analyses. 

**Code root:** `code/data_prep`

- **Generate Glasser labels** (`label.gii` and `dlabel.nii`): Glasser gifti labels are be used for parcellation and visualization. The region labels from the [original file](https://balsa.wustl.edu/78X3) were ordered as right (1-180), then left (181-360). They are reindexed in `prep_glasser_labels.sh` to be consistent with other data used in analyses, where left hemisphere regions are indices 1-180, right hemisphere regions are indices 181-360. This script calls `remap_labels.sh`. Run this as:
    
    ```bash
    cd /Users/joelleba/tractmaps/code/data_prep
    bash prep_glasser_labels.sh
    ```
    
- **Create the tract-to-region probabilities dataframe:** the tract-to-region probabilities and region names are reformatted for analysis in `prep_tract_probabilities.py`
- **Regional coordinates**: the x-y-z coordinates of Glasser regions are generated in `prep_glasser_coords.py`
- **Cognitive terms dataframe**: Cognitive maps from neurosynth are obtained using the NiMARE module and parcellated in Glasser regions using `parcellate_neurosynth.py`. Note: for this script to run, first create a yaml file containing your email address:
    
    ```yaml
    # Configuration file for neurosynth data preparation
    
    # Email address for downloading abstracts from neurosynth
    email_address: 'email.address@here.com'
    
    ```
    
- **Biological cortical properties dataframe**: The cortical maps used in the project are obtained through neuromaps and BigBrain, then parcellated in `parcellate_neuromaps.py`
- **Euclidean distances**: Regional pairwise Euclidean distances, as well as tract mean Euclidean distance, are generated in `tract_euc_distance.py`
- **Geodesic distance (sensitivity analysis)**: Regional Geodesic distances, as well as tract mean geodesic distance, are generated in `tract_geodesic_distance.py`Note that this script takes a little while to run.
- **Sensorimotor-to-association axis**: Tract S-A axis ranges are generated in `prep_sa_axis.py`
- **Tract biological cortical similarity**: The cortical similarity of tracts based on neurobiological (neuromaps) properties is generated in `tract_cortical_similarity.py`
- **Regional nulls (spin-based)**: Spin-based nulls (for Partial Least Squares analysis) are generated in`compute_nulls.py`Note: this will take a little while to run. This will output 10,000 null indices for Glasser regions.
- **Network rewiring nulls (sensitivity analysis)**: Tract rewiring nulls are generated in`tract_rewiring_nulls.py`. Note: this will take a little while to run. This will output 10,000 dataframes containing tract probabilities for rewired null tracts, saved in a pkl.

These scripts generate all the data contained in the `data/derivatives` folder. This data can be used to run the analyses and generate the manuscript figures below. 

## Analysis

### Data structure (Figure 1)

Plots for Figure 1 showing the input data structure for tract probabilities, cognitive terms, and cortical properties are generated in: 

**Path:** `code/analysis/1_data_structure/`

- `data_structure_plotting.py`  - plots the tract-to-region, cognitive terms, and biological properties matrices along with a few example maps shown on the cortical surface.
- `tracts_vis_table_1.py` - generates tract visualizations in glass brains for Table 1.

### Spatial embedding (Figure 2)

The association between the mean Euclidean distance and S-A range of tracts is examined in:

**Path:** `code/2_spatial_embedding/`

- `plot_tract_distances.py` - creates heatmaps of region coordinates and Euclidean distances with tract overlays.
- `plot_example_tract_sa_ranks.py` - plots the full S-A axis and the S-A ranks of example tracts on the brain surface.
- `spatial_embedding.py` - significance testing of the association between tract mean Euclidean distance and S-A range, and generates the correlation figure.

**Sensitivity analyses**

- `spatial_embedding_geodesic.py` - uses geodesic distance instead of Euclidean distance.
- `spatial_embedding_tract_subsets.py` - analyzes association tracts and projection tracts separately.
- `spatial_embedding_rewiring_null.py` - significance testing using tract rewiring (this takes some time).

### Terms-tracts partial least squares (Figure 3)

Partial least squares analysis to identify dominant patterns of covariance between cognitive terms and tracts is performed in:

**Path:** `/code/3_pls/`

- `pls_diagram.py` - generates plots with simulated data for the explanatory diagram.
- `pls_terms_tracts.py` - performs the PLS analysis. Note that the significance testing and cross-validation performed here takes a while to run.
- `pls_plotting.py` - plots the PLS results. The bootstrapping done for the term and tract loadings here takes a little while.

### Functional decoding (Figure 4)

The cognitive term contributions are generated for each tract in:

**Path:** `/code/4_functional_decoding/`

Code:

- `tract_term_contributions.py` - calculates the cognitive term contributions for each tract as a terms x tracts matrix containing the mean of normalized s-crores (across connected regions) per tract. Term z-scores are normalized using `scaled_robust_sigmoid` (to handle extreme values in a subset of terms, which can bias Gini coefficients) and thresholded at >1.64.
- `tract_cog_functions_plotting.py` - visualizes the tract-term association results (barplots for terms and categories, as well as word clouds)

### Tract functional diversity (Figure 5)

Tract gini coefficients of functional diversity and association with S-A range is done in:

**Path:** `/code/5_functional_diversity`

Code: 

- `tract_gini_coefficients.py` - generates Gini coefficients of diversity for each tract.
- `tract_gini_plotting.py` - plots the Lorenz curves and Gini coefficients as lollipops and in a glass brain.
- `tract_diversity_sa_axis.py` - significance testing of the association between tract Gini coefficients and S-A ranges.

**Sensitivity analyses**

- `tract_diversity_sa_axis_tract_types.py` - analyzes association tracts and projection tracts separately.
- `tract_gini_sa_correlation_rewiring_null.py` - significance testing using tract rewiring (takes some time).

### Biological cortical similarity (Figure 6)

The association between tract mean cortical similarity (based on neurobiological cortical features from neuromaps and BigBrain), S-A range and Gini coefficient of diversity is done in: 

**Path:** `analysis/6_cortical_similarity/`

Code: 

- `similarity_diagram.py` - generates plots for the explanatory diagram.
- `cortical_similarity_sa_axis_func_diversity.py` - significance testing of the correlations between tract mean cortical similarity, S-A range, and Gini coefficients of diversity.

**Sensitivity analyses**

- `cortical_similarity_tract_types.py` - analyzes association tracts and projection tracts separately.
- `gini_cortical_similarity_rewiring_null.py` - significance testing for the correlation between tract mean cortical similarity and Gini coefficients using tract rewiring (takes some time).
- `sa_range_cortical_similarity_rewiring_null.py` - significance testing for the correlation between tract mean cortical similarity and Gini coefficients using tract rewiring (takes some time).

### Individual-level analysis (Figure 7)

This section performs individual-level age and cognition GAMs, as well as associations with tract S-A range and Gini coefficients. 

**Step 1: pull qsirecon data on CUBIC**

This first step happens on CUBIC: 

**CUBIC path:** `/cbic/projects/tractmaps/code`

This code is also available in the repository, under `code/get_data`. First, pulling the data is done with scripts in the CUBIC project directory: 

- `get_subjects_list.sh` - lists all the subjects with qsirecon data. This saves out a text file in the same directory: `pnc_subject_list.txt`, which will be needed in the next scripts.
- `unzip_files.sh` - code that unzips all participant files. Thanks to Tien Tong for providing this script!
- `run_unzip_pnc_cubic.sh` - contains the file pattern and subject list for file file extraction. This script calls `unzip_files.sh` to actually extract the files. Heads up: this takes a little while. Run it with:
    
    ```bash
    bash run_unzip_pnc_cubic.sh
    ```
    
- `check_subjects.sh`  - finds which subjects where in the original `pnc_subject_list.txt` (aka, they have a qsirecon zip file), but don’t have a tsv output file. This should be 0 (all were unzipped correctly).

**Step 2: create participants-by-measures csvs for downstream analyses**

**Local path:** `analysis/7_individual_level`

- `group_level_tract_scalars_pnc.R` - saves a csv with FA values in `data/derivatives/individual_level_pnc/cleaned` . These will be used for final sample selection below. Note that it will take a while to load all subjects’ files.
- `group_level_qc_measures_pnc.R` - generates a csv with dMRI QC measures. This also takes a while.
- `sample_creation_pnc.py` - applies data exclusion as done in Audrey’s [paper](https://github.com/PennLINC/luo_wm_dev/blob/main/code/sample_construction/construct_initial_sample/PNC_InitialSampleSelection.Rmd). This outputs a final sample csv in `data/derivatives/individual_level_pnc/final_sample` . This will be used in downstream analyses.

**Step 3: run GAMs**

**Local path:** `analysis/7_individual_level`

- `scpt_GAM_tractmaps_pnc.R` - runs GAMs on each tract to determine the relationship between tract FA and age, as well as cognition. This outputs partial R2 and FDR-corrected p-values in: `results/individual_level/`
- `func_GAM_tractmaps.R` - is called by `scpt_GAM_tractmaps_pnc.R` to fit the GAMs.

**Step 4: association between age effects, cognition effects, and tract properties**

**Local path:** `analysis/7_individual_level`

Code: 

- `partial_r2_pnc_plotting.py` - loads and plot partial R² results from individual-level GAM analyses. Plots correlation showing the relationship between tract properties (Gini coefficient and S-A range) and partial R² from age and cognition GAMs.

Et voilà! 😊

## A note on utilities

The repository also contains some utility functions in the `code/utils` folder. These get called throughout the data preparation and analyses steps. Here is a brief description: 

- `figure_formatting.py` - takes care of formatting all figures generated in the manuscript, e.g. ensuring consistent font size, font type, etc.
- `matrix_to_tracts.py` - function that extracts pairwise values from any inter-regional matrix.
- `tm_utils.py` - contains functions for loading and saving data; plotting brain maps on brain surfaces; and plotting correlations along with permutation testing.
- `tract_visualizer.py` - code to plot tracts in glass brains using DSI studio (through the command line). This can plot tracts all together or separately in a glass brain, can take specific color inputs or color gradients based on an continuous tract metric (ex: tract S-A range), and can save tracts in different layouts (medial, lateral, grid of tracts, etc.). Note that it requires access to tract trk files for plotting (for instance, this project used publicly available trk files at [https://github.com/data-others/atlas/releases/download/hcp1065/hcp1065_avg_tracts_trk.zip](https://github.com/data-others/atlas/releases/download/hcp1065/hcp1065_avg_tracts_trk.zip)). It also requires the `abbreviations.xlsx` for tract name mapping.
- `tract_visualizer_quickstart.py` - contains a few examples illustrating how to use `tract_visualizer.py` to plot tracts.
- `tract_visualizer_readme.md` - a brief description of tract visualization scripts, their key features, and requirements.