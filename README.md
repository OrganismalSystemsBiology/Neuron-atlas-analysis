
## Advanced Analyses

3. **Calculation of local cell density consistency after registration**  
**Notebook:** `/script/3_local_cell_density_consistency.ipynb`

4. **Whole-brain regional plots**  
**Notebook:** `/script/4_Whole_brain_regional_plots.ipynb`

5. **Age-dependent linear regression analysis**  
- **Notebook (Regional):** `/script/5-1_Regional_Age-dependent_linear_regression_analysis.ipynb`  
- **Notebook (Single Cell):** `/script/5-2_Sigle_Cell_level_Age-dependent_linear_regression_analysis.ipynb`

6. **Onset analysis**  
- **Notebook (Regional):** `/script/6-1_Regional_Onset_analysis.ipynb`  
- **Notebook (Single Cell):** `/script/6-2_Sigle_Cell_level_Onset_analysis.ipynb`

7. **Pathological pseudotime analysis**  
- **Notebook (Aβ):** `/script/7-1_Aβ_Pathological_pseudotime_analysis.ipynb`  
- **Notebook (Microglia):** `/script/7-2_microglial_Pathological_pseudotime_analysis.ipynb`  
- **Notebook (Microgliosis):** `/script/7-3_microgliosis(NND)_Pathological_pseudotime_analysis.ipynb`

8. **Thal phase analysis**  
**Notebook:** `/script/8_Thal_phase_analysis.ipynb`

9. **Spatial density variation analysis (3D vs. 2D) and linear regression for biomarker 2D recall**  
**Notebook:** `/script/9_3D_vs_2D_analysis.ipynb`

10. **Microglial 3D morphology and microgliosis analysis**  
 **Notebook:** `/script/10_Microglial_3D_morphology_and_microgliosis_analysis.ipynb`

11. **Microglial spatial gradient analysis using pGM/pWM masks and intermediate segmentations**  
 **Notebook:** `/script/11_Microglial_spatial_gradient_analysis.ipynb`

12. **Spatial single-cell-level risk analysis and spatial transcriptome integration analysis**  
 - **Notebook (Risk Analysis):** `/script/12-1_Spatiotemporal_Neuron_Microglia_risk_analysis.ipynb`  
 - **Notebook (Transcriptome):** `/script/12-2_Transcritional_Neuron_Microglia_risk_analysis.ipynb`

## Summary of Results

1. **Statistical Analysis Summary (Figs and Extended Data Figs)**  
Refer to **"Supplementary Table 1. Summary of statical analysis.xlsx"** for a detailed breakdown.

### Available Analyzed Data for Download

- B6J Wild-type (8 weeks old, 1,3,5,7,9,12 months, male), APPNL-G-F model (1,3,5,7,9 months, male), VCP model (8-9 weeks old, male), and TMT model (8 weeks old, male) cell data (xyz, cell type, atlas annotation ID) and template images used for registration.  
- Aβ plaque data (xyz coordinates, plaque size, plaque intensity, atlas annotation ID) for B6J WT (1,3,5,7,9 months, male) and APPNL-G-F model (1,3,5,7,9 months, male), following methods described in Yanai et al., *Brain Communications* 2024 ([Tau-analysis repo](https://github.com/OrganismalSystemsBiology/Tau-analysis.git)).

**Please download from:**  
[Google Drive](https://drive.google.com/drive/folders/1XrRgaWScrQQk3uV722mXu4JfQgIKu4IZ)

> **Note:** This code does not currently include source image data from light-sheet imaging.

## System Requirements

Tested under the following conditions (versions chosen as required for code compatibility):

- **CentOS Linux release 7.9.2009 (Core)** with Python 3.6.8 or Python 3.9.0 in a virtualenv  
- **Ubuntu 22.04.4 LTS** with Python 3.7.17 or Python 3.9.19 in a virtualenv  

## Citation

If you utilize this code in your research, please cite our paper:

**Whole-Brain Single-Neuron Atlas Reveals Microglial Security Hole Accelerating Neuronal Vulnerability**  
Mitani T.T. et al.  
DOI: to be updated
