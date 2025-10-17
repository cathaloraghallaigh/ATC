# From Pockets to Products  
### A Data-Driven Framework for Classification of Monoterpene Synthase Product Preferences

**Author:** Cathal Ó Raghallaigh  
**Affiliation:** Manchester Institute of Biotechnology, The University of Manchester  
**Preprint:** [ChemRxiv, 2025](https://doi.org/10.26434/chemrxiv-2025-mf4xs)

<img width="4678" height="1709" alt="WorkflowFigure0916" src="https://github.com/user-attachments/assets/27b563a6-022d-481a-9f1b-e444d9abb905" />

---

## Overview
This repository provides all datasets and notebooks used in the ChemRxiv preprint *“From Pockets to Products: A Data-Driven Framework for Classification of Monoterpene Synthase Product Preferences.”*  
The project integrates **structural bioinformatics** and **machine learning** to predict whether monoterpene synthases produce **linear or cyclic** products based on sequence- and structure-derived features.

---

## Notebooks and Datasets

 Notebook / Dataset | Description | Includes |
|--------------------|--------------|-----------|
| `LOOXGBoost.ipynb` | Leave-one-out cross-validation for the main XGBoost classification model. Implements stratified LOO-CV to assess model robustness and generalisation. | XGBoost model, accuracy metrics, confusion matrix, per-class performance. |
| `BOXGBoost.ipynb` | Bayesian optimisation of XGBoost hyperparameters and feature group subsets for binary classification (linear vs cyclic). Evaluates combinations of structure-derived features to optimise accuracy and interpret key contributors.| Feature subgroup breakdown, group-wise addition, binary accuracy tracking, convergence plots. |
| `TestOnPluskalDataset.ipynb` | External validation using the dataset from **Samusevich *et al.* (2024)**, *Structure-enabled enzyme function prediction unveils elusive terpenoid biosynthesis in archaea* ([bioRxiv, 2024.01.29.577750](https://doi.org/10.1101/2024.01.29.577750)). |Benchmarking of model prediction against unseen data from the Pluskal Lab. |
| `StrucResCons.ipynb` | Performs structure-based residue conservation analysis by aligning multiple protein structures to a reference model and identifying corresponding residues based on spatial proximity. This enables comparison of active-site architecture across proteins independent of sequence alignment, providing insights into conserved structural motifs and residue distributions associated with catalytic or binding functions. | Structural alignment, spatial residue matching, conservation scoring, sequence logo generation, and per-protein mapping tables. |
| `StatsScript.ipynb` | Statistical comparisons of feature distributions between enzyme classes. | Two-tailed t-tests, feature significance ranking, visual summaries. |
| `FullDataset.csv` | Comprehensive main dataset used within project, of 88 experimentally characterised monoterpene synthases with active site structurally derived features. | Raw features used for model training and feature selection. |
| `TestTrainSet.csv` | Training and test partition used for supervised model benchmarking. | 80/20 split for reproducible model comparison. |
| `ValidationSet.csv` | Independent validation dataset containing enzymes not used in training or optimisation. | Used for unbiased evaluation of model generalisation. |
| `CharacterisedSeqRef.csv` | Reference table of experimentally characterised sequences with functional annotation. | UniProt/GenBank accessions, enzyme names, and product class (linear/cyclic). |
| `PreprintExtraLinLim.csv` | Supplementary dataset containing **additional linalool and limonene synthases** identified in **Samusevich *et al.* (2024)**. | Used for external validation and expanded chemical space assessment. |
| `PreprintExtraLinLimData.csv` | Processed structural and physicochemical data corresponding to the above enzymes. | Extended feature-level validation set. |

---

## Quick Start

### 1. Create Environment
```bash
conda env create -f environment.yml
conda activate mtsenv
```
## Citation

If you use this repository or data, please cite:

Ó Raghallaigh, C. (2025). From Pockets to Products: A Data-Driven Framework for Classification of Monoterpene Synthase Product Preferences. ChemRxiv. https://doi.org/10.26434/chemrxiv-2025-mf4xs
