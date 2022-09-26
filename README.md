# NLTI Analysis of Lai et al.

[![DOI](https://zenodo.org/badge/489268188.svg)](https://zenodo.org/badge/latestdoi/489268188)

This repository contains the software used to produce the analysis results presented in the article (Not published yet).

To reproduce the analysis perform the following steps:

## Prerequisite
You need a Linux or Mac system with the [conda package manager](https://docs.conda.io/en/latest/miniconda.html) installed.

## Create environment
To create the conda environment, run the following command from within the cloned repo directory
```bash
conda env create -n nlti --file environment.yml
```

## Start Jupyter lab
Activate the newly created environment and start Jupyter lab
```bash
conda activate nlti
jupyter lab
```

## Download the data
Open the notebook `Get_data.ipynb` and execute all cells.

## Run Analysis
Execute all notebooks matching `Fig*.ipynb` in the order indicated by the numbering.

## Create plots
Execute all notebooks matching `plot_Fig*.ipynb`.
