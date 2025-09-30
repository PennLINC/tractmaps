# Tract Visualizer Example Scripts

This directory contains example scripts demonstrating how to use the `TractVisualizer` class for creating colored tract visualizations using DSI Studio.

## Files


### 1. `tract_visualizer_quickstart.py`
**For new users** - Contains simple, essential examples to get started quickly:
- Basic data visualization with your own metrics
- Single color visualizations
- All tracts visualization
- Gradient plots


### 2. `tract_visualizer.py`
The main TractVisualizer class with full documentation.


## Key Features

- **Multiple input formats**: DataFrame with values, tract lists, single colors, color lists
- **Flexible coloring**: Value-based colormaps, custom colors, single colors
- **View options**: Lateral, medial, or grid layouts
- **Plot modes**: Individual tracts, all tracts together, gradient plots
- **Colorbar support**: Automatic legends and colorbars
- **Tract name matching**: Supports multiple tract naming conventions

## Requirements

- DSI Studio installed and accessible (this code was tested using DSI studio version Hou "侯" May 27 2025)
- Tract files (.trk.gz format) in proper directory structure
- `abbreviations.xlsx` file for tract name mapping
- Python packages: pandas, numpy, matplotlib, PIL


## Output

Visualizations are saved as JPEG images in the output directory. Default location:


You can specify custom output directories using the `output_dir` parameter.

## Troubleshooting

- Ensure DSI Studio is installed and the path in the script matches your installation
- Check that tract files exist in the expected directory structure
- Verify the `abbreviations.xlsx` file is present for tract name matching
- Use `validate_tract_names()` method to check tract name validity

For detailed documentation, see the docstrings in `tract_visualizer.py`.
