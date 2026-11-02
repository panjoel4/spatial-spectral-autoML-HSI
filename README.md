# Efficient and Scalable Mineralogical Classification of Hyperspectral Drill-Core Imagery Using Spatial-Spectral Patch-Based AutoML

Journal: Computers &amp; Geosciences  
Authors: Izzul Qudsi, Hilary Corlett, Khalid Al-Ramadan, Abduljamiu Olalekan Amao, Israa S. Abu-Mahfouz, Cédric M. John, Ardiansyah Koeshidayatullah  
Corresponding author: a.koeshidayatullah@kfupm.edu.sa  

## Overview

This repository contains the workflow used in the manuscript on a jupyter notebook file with several helper (py files). Follow the step by step guide on the jupyter notebook to reproduce the research using your dataset.

## Repository contents

- `Spatial-spectra_patch_AutoML.ipynb` — configuration and end-to-end analysis workflow.
- `data_tools.py` — HSI data loading and pre-processing.
- `patch_tools.py` — patch selection and visualisation.
- `vae_tools.py` — VAE training, pixel clustering, cluster summaries, and spectral plots.
- `model_tools.py` — AutoGluon training, image-wide prediction, metrics, and core-stick analysis.
- `requirements.txt` — Python package dependencies.

## Installation

Use Python 3.9–3.12 (Python 3.9 is the compatibility baseline used for this setup). Install the project dependencies as follows:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include TensorFlow and AutoGluon as well as JupyterLab; installation can take several minutes and requires a compatible Python version and platform.
