# Pockets to Products  
### A Data-Driven Framework for Classification of Monoterpene Synthase Product Preferences
**Authors:** Cathal Ó Raghallaigh, Nigel S. Scrutton, Sam Hay  
**Affiliation:** Manchester Institute of Biotechnology, The University of Manchester 

[![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2025--mf4xs-brightgreen.svg)](https://doi.org/10.26434/chemrxiv-2025-mf4xs)


<img width="3862" height="1396" alt="Picture8" src="https://github.com/user-attachments/assets/52767163-5c5c-4168-a85f-185295d26b69" />

## Overview  
This repository provides all datasets and notebooks used in the ChemRxiv preprint *“From Pockets to Products: A Data-Driven Framework for Classification of Monoterpene Synthase Product Preferences.”*  
The project integrates **structural bioinformatics** and **machine learning** to predict whether monoterpene synthases produce **linear or cyclic** products using sequence- and structure-derived features.  

---

## Notebooks and Datasets  

## Notebooks and Datasets

| Notebook / Dataset                                 | Description                                                                                                                         | Includes                                                                                                         |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `CharacterisedSeqRef.csv`                          | Reference table of characterised sequences.                                                                                         | UniProt/GenBank accessions, enzyme names and product class.                                                      |
| `FullDataset.csv`                                  | Full dataset of 87 characterised monoterpene synthases with active-site features.                                                   | Features used for model training, validation and interpretation.                                                 |
| `TestTrainSet.csv`                                 | Training partition used for model development.                                                                                      | 69 enzymes from the full dataset.                                                                                |
| `ValidationSet.csv`                                | Independent validation dataset used to assess generalisation.                                                                       | 18 enzymes from the full dataset.                                                                                |
| `StatsScript.ipynb`                                | Statistical comparison of feature distributions between classes.                                                                    | t-tests, significance ranking and visual summaries.                                                              |
| `StrucResCons.ipynb`                               | Structure-based residue conservation analysis. Aligns models to a reference and identifies spatially corresponding residues.        | Structural alignment, conservation scoring, residue mapping and logo generation.                                 |
| `BOXGBoost.ipynb`                                  | Bayesian optimisation of XGBoost hyperparameters and feature-group subsets.                                                         | Feature-group selection, accuracy tracking and convergence plots.                                                |
| `BayesianOut.csv`                                  | Output from Bayesian optimisation for feature-set reduction.                                                                        | Model performance and selected feature groups.                                                                   |
| `LOOXGBoost.ipynb`                                 | Leave-one-out evaluation of the selected XGBoost model on the training partition.                                                   | Accuracy, confusion matrix, misclassified enzymes and feature importance.                                        |
| `TestTrainEval.ipynb`                              | Trains the FS1 XGBoost model on `TestTrainSet.csv` and evaluates it on `ValidationSet.csv`.                                         | Accuracy, balanced accuracy, MCC, per-class metrics, confusion matrix and feature importance.                    |
| `SHAP_and_Gain_Analysis.ipynb`                     | Repeated stratified five-fold evaluation and interpretation of the FS1 model using the full dataset.                                | Confusion matrix, XGBoost gain and SHAP figures.                                                                 |
| `PreprintExtraLinLim.csv`                          | Additional linalool and limonene synthases from **Samusevich *et al.* (2024)**.                                                     | External validation sequences and product classes.                                                               |
| `PreprintExtraLinLimData.csv`                      | Processed structural and physicochemical data for the above enzymes.                                                                | Active-site features used for external validation.                                                               |
| `TestOnPluskalDataset.ipynb`                       | External validation using **Samusevich *et al.* (2024)**, *bioRxiv* [2024.01.29.577750](https://doi.org/10.1101/2024.01.29.577750). | Evaluation against an independent Pluskal Lab dataset.                                                           |
| `Structures/`                                      | Structures of the monoterpene synthases used in the study.                                                                          | PDB models and active-site pockets generated using FPocket.                                                      |
| `SupplementaryInformation/LOCO/`                   | Leave-one-cluster-out validation using published sequence, structural and all-active-site-feature clusters.                         | Validation and split-generation notebooks, published split manifests and the random five-fold manifest.          |
| `SupplementaryInformation/ModelGeneralisability/`  | Tests the model and active-site classification workflow across wider terpene space.                                                 | Wider-monoterpene and sesquiterpene datasets and notebooks.                                                      |
| `SupplementaryInformation/Structural_Sensitivity/` | Tests the sensitivity of active-site features and predictions to AlphaFold model rank.                                              | Structural sensitivity dataset and analysis notebook.                                                            |
| `reproduce_results.py`                             | Runs the main model analyses using the supplied datasets and published LOCO splits.                                                 | Test-train evaluation, leave-one-out validation, SHAP and gain analysis, Pluskal validation and LOCO validation. |

---

## Installation Options

**Option 1: Conda**

```bash
conda env create -f environment.yaml
conda activate mtsenv
python reproduce_results.py
```

**Option 2: Pip**

Use Python 3.10, then run:

```bash
python -m pip install -r requirements.txt
python reproduce_results.py
```

The wider-monoterpene, sesquiterpene and structural-sensitivity notebooks are run separately from their respective supplementary-information folders. MMseqs2 and Foldseek are required only to regenerate the LOCO clusters and splits; they are not required to run validation using the published split files.

## Citation

If you use this repository or data, please cite:

Ó Raghallaigh, C. (2025). From Pockets to Products: A Data-Driven Framework for Classification of Monoterpene Synthase Product Preferences. ChemRxiv. https://doi.org/10.26434/chemrxiv-2025-mf4xs

## License

This project is licensed under the MIT License. You are free to use, modify and distribute this work with appropriate attribution. See the [LICENSE](./LICENSE) file for full terms.


