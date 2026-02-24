# analysis-codes-ml
This is the set of scripts and outputs (including data preparation for microbial diversity and pollutant sources, as well as the primary machine learning analysis) in Python used in our study.

Study title: Identifying Nonlinear Thresholds in Aquatic Microbial Diversity Along Pollution Gradients  
Overview:
This repository contains the scripts used to analyze microbial diversity relative to environmental pollution using publicly available genomic and georeferenced pollution datasets. The analysis investigates whether changes in microbial diversity follow linear trends or show non-linear behavior across increasing pollution levels. The code is intended to support reproducibility of the data processing, analysis, and visualization pipeline used in the associated study.

Repository Structure:

data/ - directory for raw input datasets (not included due to size and licensing).  
└── .gitkeep  
outputs/ – Directory for generated plots and processed tables.  
LICENSE.md  
README.md  
analysis_and_plots.py – Performs statistical analysis and generates diversity vs pollution visualizations.  
microbial_diversity.py – Computes microbial diversity indices from processed genomic data.  
pollution_sources.py – Integrates and cleans genomic and pollution site datasets.  
requirements.txt - Install the required modules from this file.

Data Sources:

The analysis uses publicly available datasets describing:  
Microbial genomic sequences and taxonomic information - μSudAqua.  
Georeferenced pollution or contamination site data - OpenStreetMap and Overpass API (queried).  
Please download the dataset separately before running the scripts.  

Method Summary:
Genomic and pollution datasets are cleaned and spatially matched.
Microbial diversity metrics are calculated for each site.
Diversity values are analyzed against pollution intensity.
Results are visualized as diversity–pollution relationships.

Usage:
Install required Python dependencies listed in requirements.txt with "pip install -r requirements.txt".

Place downloaded datasets in the data/ directory.

Run the scripts in the following order:

microbial_diversity.py,
pollution_sources.py,
analysis_and_plots.py.

Outputs - 
The pipeline produces:
Processed tables of microbial diversity and pollution levels.
Plots showing microbial diversity as a function of pollution intensity.

Limitations:
This analysis is constrained by the resolution and geographic coverage of available datasets. The results describe statistical relationships and do not establish causal mechanisms. Interpretation of patterns should consider potential confounding environmental factors.

Citation:
If this code is used for academic or educational purposes, please cite the associated study and the original data sources.

License:
This project is released under the MIT License.
