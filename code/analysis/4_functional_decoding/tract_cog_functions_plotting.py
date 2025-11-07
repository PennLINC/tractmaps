# ------------------------------------------------------------------------------------------------
# --- Plotting tract-term association results ---
# ------------------------------------------------------------------------------------------------

# This script visualizes the tract-term association results from tract_term_contributions.py
# It creates barplots of terms and categories, and word clouds.
# Uses tract_term_contributions.csv which is a terms x tracts matrix containing the mean of normalized z-scores (across connected regions) per tract.

# ------------------------------------------------------------------------------------------------
# --- Load packages ---
# ------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from utils.tract_visualizer import TractVisualizer
from utils.figure_formatting import setup_figure, save_figure
# ------------------------------------------------------------------------------------------------
# --- Set up inputs and outputs ---
# ------------------------------------------------------------------------------------------------

# Set root
root_dir = '/Users/joelleba/PennLINC/tractmaps'

# Set up directories
results_dir = os.path.join(root_dir, 'results/tract_functional_decoding/decoding_figures')
term_barplots_dir = os.path.join(results_dir, 'term_barplots')
category_barplots_dir = os.path.join(results_dir, 'category_barplots')
wordclouds_dir = os.path.join(results_dir, 'wordclouds')

# Create output directories
for directory in [term_barplots_dir, category_barplots_dir, wordclouds_dir]:
    os.makedirs(directory, exist_ok=True)

# Define cognitive categories and their colors
cat_colors_dict = {
    'action': '#35b2d4',  # blue 
    'exec./cog.\ncontrol': '#41c899',  # sea green 
    'decis. making': '#8fd6b9',  # light green
    'language': '#66B2FF',  # light blue
    'attention': '#9999FF',  # light purple 
    'perception': '#CCCCFF',  # light lavender
    'other': '#d48ede',  # light magenta
    'motivation': '#FF99CC',  # pink
    'social function': '#f65690',  # light red 
    'learning/memory': '#fc9005',  # light orange
    'emotion': '#ff4c33',  # medium red
}

# ------------------------------------------------------------------------------------------------
# --- Load data ---
# ------------------------------------------------------------------------------------------------

# Load the tract-term contributions matrix
contributions_path = os.path.join(root_dir, 'results/tract_functional_decoding/decoding_results/tract_term_contributions.csv')
contributions_matrix = pd.read_csv(contributions_path, index_col=0)

# Load term to category mapping
terms_to_cats = pd.read_csv(os.path.join(root_dir, 'data/raw/neurosynth_categories/neurosynth_categories_names_125.csv'))
terms_to_cats = dict(zip(terms_to_cats['cog_term'], terms_to_cats['cog_category']))

# Rename terms in matrix (replace spaces with underscores) to match category mapping
contributions_matrix.index = contributions_matrix.index.str.replace(' ', '_')

# ------------------------------------------------------------------------------------------------
# --- Compare terms between datasets ---
# ------------------------------------------------------------------------------------------------

# Compare terms between datasets
matrix_terms = set(contributions_matrix.index)
category_terms = set(terms_to_cats.keys())

terms_only_in_matrix = matrix_terms - category_terms
terms_only_in_categories = category_terms - matrix_terms

print("\nTerms in matrix but not in category mapping:")
for term in sorted(terms_only_in_matrix):
    print(f"- {term}")

print("\nTerms in category mapping but not in matrix:")
for term in sorted(terms_only_in_categories):
    print(f"- {term}")

print(f"\nTotal terms in matrix: {len(matrix_terms)}")
print(f"Total terms in category mapping: {len(category_terms)}")
print(f"Terms in both: {len(matrix_terms.intersection(category_terms))}")

# Calculate global maximum contributions for consistent y-axis limits
# Calculate actual maximum contribution from all terms
contributions_max = contributions_matrix.max().max()  # Get the maximum value across all terms
contributions_y_min = 0  # Since we only show positive contributions
contributions_y_buffer = contributions_max * 0.1  # Add 10% buffer to the top

# Calculate global maximum for category means
category_means_all = []
for tract in contributions_matrix.columns:
    tract_contributions = contributions_matrix[tract]
    term_data = pd.DataFrame({
        'term': tract_contributions.index,
        'contribution': tract_contributions.values
    })
    term_data['category'] = term_data['term'].map(terms_to_cats)
    category_means = term_data.groupby('category')['contribution'].mean()
    category_means_all.extend(category_means.values)

categories_max = max(category_means_all) if category_means_all else 0
categories_y_min = 0
categories_y_buffer = categories_max * 0.1  # Add 10% buffer to the top

# Print the maximum values for reference
print(f"\nMaximum values for y-axis limits:")
print(f"Terms barplot: {contributions_max:.2f} (calculated from all terms)")
print(f"Categories barplot: {categories_max:.2f} (calculated from all category means)")


# ------------------------------------------------------------------------------------------------
# --- Plotting functions for term barplots, category barplots, and word clouds ---
# ------------------------------------------------------------------------------------------------

def plot_tract_terms(
    tract_name, contributions_matrix, terms_y_max, terms_y_buffer, 
    terms='all_terms'
):
    """
    Create term barplot for a single tract.

    Parameters:
    -----------
    tract_name : str
        Name of the tract to visualize
    contributions_matrix : pandas.DataFrame
        Matrix of contribution scores
    terms_y_max : float
        Maximum y-axis value for term barplot
    terms_y_buffer : float
        Buffer to add to terms_y_max for better visualization
    terms : str
        'all_terms' (default) or 'nonzero_only'
    """
    # Get term data for this tract
    tract_contributions = contributions_matrix[tract_name]

    term_data = pd.DataFrame({
        'term': tract_contributions.index,
        'contribution': tract_contributions.values
    })

    if terms == 'nonzero_only':
        # Filter for nonzero terms only
        term_data_plot = term_data[term_data['contribution'] > 0].copy()
    elif terms == 'all_terms':
        # Keep all terms, including zeros
        term_data_plot = term_data.copy()
    else:
        raise ValueError("terms argument must be 'all_terms' or 'nonzero_only'")

    if len(term_data_plot) == 0:
        print(f"No terms to plot for {tract_name} with terms='{terms}'")
        return

    # Add category information
    term_data_plot['category'] = term_data_plot['term'].map(terms_to_cats)
    ordered_categories = list(cat_colors_dict.keys())
    term_data_plot['category'] = pd.Categorical(term_data_plot['category'], categories=ordered_categories, ordered=True)
    term_data_plot = term_data_plot.sort_values(['category', 'contribution'], ascending=[True, False])
    
    # Replace underscores with spaces for pretty display
    term_data_plot['term_display'] = term_data_plot['term'].str.replace('_', ' ')

    tract_max_contrib = term_data_plot['contribution'].max()
    tract_y_max = max(terms_y_max, tract_max_contrib)

    # Determine output directory based on 'terms' argument
    if terms == 'all_terms':
        output_dir = term_barplots_dir
    else:
        output_dir = f'{term_barplots_dir}_nonzero'
        os.makedirs(output_dir, exist_ok=True)

    # Set figure size based on 'terms' argument (convert inches to mm)
    if terms == 'all_terms':
        fig_width_mm = 178 # mm
    else:
        fig_width_mm = 95 # mm
    fig_height_mm = 48 # mm

    fig, ax = setup_figure(width_mm=fig_width_mm, height_mm=fig_height_mm, 
                           margins_mm=(12, 0, 31, 0), base_pt=5) # smaller font size for terms barplot
    sns.barplot(
        x='term_display', y='contribution', data=term_data_plot, hue='category',
        palette=cat_colors_dict, hue_order=ordered_categories, ax=ax
    )
    ax.set_ylabel('Mean term\ncontribution')
    ax.set_xlabel('')
    ax.legend().set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_ylim(contributions_y_min, tract_y_max + terms_y_buffer)
    
    # Set y-ticks every 0.2
    y_min, y_max = ax.get_ylim()
    y_ticks = np.arange(0, y_max, 0.2)
    ax.set_yticks(y_ticks)

    # Set x-ticks to black
    for tick in ax.get_xticklabels():
        tick.set_color('black')

    # Add background boxes and dotted lines to separate cognitive categories
    category_boundaries = []
    current_category = None
    
    for i, category in enumerate(term_data_plot['category']):
        if current_category is not None and category != current_category:
            # Add boundary line between categories
            category_boundaries.append(i - 0.5)
        current_category = category
    
    # Add transparent background boxes for each category
    # Create proper category ranges
    unique_categories = term_data_plot['category'].unique()
    
    for category in unique_categories:
        # Find all indices for this category
        category_indices = term_data_plot[term_data_plot['category'] == category].index
        positions = [term_data_plot.index.get_loc(idx) for idx in category_indices]
        
        if positions:
            start_pos = min(positions)
            end_pos = max(positions)
            
            # Draw transparent box covering the full category range
            ax.axvspan(start_pos - 0.5, end_pos + 0.5, 
                       color=cat_colors_dict[category], 
                       alpha=0.2 , 
                       zorder=0)
    
    # Reset x-axis limits to prevent extra whitespace
    ax.set_xlim(-0.5, len(term_data_plot) - 0.5)
    
    # Draw vertical dotted lines at category boundaries extending to bottom of x-tick labels
    for boundary in category_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.9, linewidth=0.3, 
                   ymin=-0.45, ymax=1, clip_on=False)

    # Set axis linewidths to 0.3
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    ax.tick_params(axis='both', width=0.3)
    
    sns.despine(ax=ax)
    save_figure(fig, f'{output_dir}/barplot_terms_{tract_name}_{terms}.svg')
    plt.close(fig)


def plot_tract_categories(tract_name, contributions_matrix, categories_y_max, categories_y_buffer):
    """
    Create category barplot for a single tract by aggregating term contributions by category, with SEM error bars.
    
    Parameters:
    -----------
    tract_name : str
        Name of the tract to visualize
    contributions_matrix : pandas.DataFrame
        Matrix of contribution scores
    categories_y_max : float
        Maximum y-axis value for category barplot
    categories_y_buffer : float
        Buffer to add to categories_y_max for better visualization
    """
    # Get contributions for this tract
    tract_contributions = contributions_matrix[tract_name]
    
    # Create DataFrame with terms and their contributions
    term_data = pd.DataFrame({
        'term': tract_contributions.index,
        'contribution': tract_contributions.values
    })
    
    # Add category information
    term_data['category'] = term_data['term'].map(terms_to_cats)
    
    # Calculate mean and SEM per category
    def sem(x):
        x = x.dropna()
        return x.std() / np.sqrt(len(x)) if len(x) > 0 else np.nan
    category_means = term_data.groupby('category').agg(
        contribution=('contribution', 'mean'),
        sem=('contribution', sem)
    ).reset_index()
    
    if len(category_means) == 0:
        print(f"No category data found for {tract_name}")
        return
    
    # Create ordered category list from cat_colors_dict
    ordered_categories = list(cat_colors_dict.keys())
    
    # Ensure categories are in the correct order
    category_means['category'] = pd.Categorical(category_means['category'], 
                                             categories=ordered_categories,
                                             ordered=True)
    category_means = category_means.sort_values('category')
    
    # Calculate tract-specific category y-max
    tract_cat_max = category_means['contribution'].max()
    tract_cat_y_max = max(categories_y_max, tract_cat_max)
    y_min = 0
    
    fig, ax = setup_figure(width_mm=30, height_mm=20, margins_mm=(10, 0, 2, 2))
    sns.barplot(
        x='category',
        y='contribution',
        data=category_means,
        palette=cat_colors_dict,
        order=ordered_categories,
        hue='category',
        legend=False,
        ax=ax
    )
    # Add error bars manually (skip if SEM is all NaN)
    if 'sem' in category_means.columns and not category_means['sem'].isnull().all():
        x_coords = range(len(category_means))
        ax.errorbar(
            x=x_coords,
            y=category_means['contribution'].values,
            yerr=category_means['sem'].values,
            fmt='none',
            ecolor='black',
            capsize=0.8,
            lw=0.3,
            capthick=0.2,
            zorder=10
        )
    y_max = tract_cat_y_max + categories_y_buffer
    ax.set_ylim(y_min, y_max)
    y_min, y_max = ax.get_ylim()
    y_ticks = np.arange(0, y_max, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_ylabel('Category contrib.')
    ax.set_xlabel('')
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    ax.tick_params(axis='both', width=0.3)
    sns.despine(ax=ax)
    save_figure(fig, f'{category_barplots_dir}/barplot_categories_{tract_name}.svg')
    plt.close(fig)


def plot_tract_wordcloud(tract_name, contributions_matrix, weighting='term_contributions'):
    """
    Create word cloud for a single tract.
    
    Parameters:
    -----------
    tract_name : str
        Name of the tract to visualize
    contributions_matrix : pandas.DataFrame
        Matrix of contribution scores
    weighting : str
        'term_contributions' (default) - uses raw contributions as weights
    """
    # Get contributions for this tract
    tract_contributions = contributions_matrix[tract_name]
    
    # Create weights dictionary (only nonzero contributions)
    # Replace underscores with spaces for display
    wordcloud_weights = {}
    term_mapping = {}  # Map display name (with spaces) to original name (with underscores)
    for term, contrib in tract_contributions.items():
        if contrib > 0:
            display_term = term.replace('_', ' ')
            wordcloud_weights[display_term] = contrib
            term_mapping[display_term] = term
    
    if not wordcloud_weights:
        print(f"No nonzero contributions found for {tract_name}")
        return

    def category_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        # Look up original term name (with underscores) for category mapping
        original_term = term_mapping.get(word, word.replace(' ', '_'))
        category = terms_to_cats.get(original_term, 'other')
        return cat_colors_dict.get(category, cat_colors_dict['other'])

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_weights)
    wordcloud = wordcloud.recolor(color_func=category_color_func)

    fig, ax = setup_figure(width_mm=10*25.4, height_mm=5*25.4, margins_mm=(0, 0, 0, 0))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    save_figure(fig, f'{wordclouds_dir}/wordcloud_{tract_name}_{weighting}.svg')
    plt.close(fig)



def main():
    """
    Main function to create all visualizations.
    """
    # Create legend for cognitive categories
    fig, ax = setup_figure(width_mm=105, height_mm=20, margins_mm=(0, 0, 0, 0))
    for category, color in cat_colors_dict.items():
        ax.plot([], [], color=color, label=category, marker='s', linestyle='None', markersize=8)
    ax.axis('off')
    ax.legend(title='Cognitive category', loc='center', frameon=False, ncol=4, title_fontsize=7)
    save_figure(fig, f'{results_dir}/barplot_legend_categories.svg')
    plt.close(fig)
    
    # Process each tract with progress bar
    for tract_name in tqdm(contributions_matrix.columns, desc="Processing tracts"):
        # Create term barplot
        plot_tract_terms(tract_name, contributions_matrix, 
                        terms_y_max=contributions_max,
                        terms_y_buffer=contributions_y_buffer,
                        terms='all_terms')
        
        # Create category barplot
        plot_tract_categories(tract_name, contributions_matrix, 
                            categories_y_max=categories_max,
                            categories_y_buffer=categories_y_buffer)
        
        # Create word cloud
        plot_tract_wordcloud(tract_name, contributions_matrix)

    # Create tract visualizations using TractVisualizer
    output_dir = os.path.join(results_dir, 'tract_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    viz = TractVisualizer(root_dir=root_dir)
    example_tracts = ['AF', 'CST', 'UF', 'VOF'] # this will create both left and right tracts
    viz.visualize_tracts(tract_list=example_tracts, 
                        single_color='#626bda', 
                        plot_mode='iterative',
                        output_dir=output_dir)

# ------------------------------------------------------------------------------------------------
# --- Main function to generate all plots ---
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main() 
    print("Plots saved to: ", results_dir)