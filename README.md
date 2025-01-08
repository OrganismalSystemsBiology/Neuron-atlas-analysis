# Neuron Atlas Analysis

This repository provides code for whole-brain-scale analyses of cell types (neurons, microglia, and cell nuclei) detected from mouse brain imaging data acquired via tissue clearing and 10x light-sheet microscopy. We perform single-cell-level registration to a Neuron Atlas and implement advanced analyses including:

- Statistical evaluation of regional and single-cell distributions  
- Pathological pseudotime analysis  
- Spatial transcriptome integration  
- Spatial single-cell risk analysis  

## Data Specifications

- **Voxel resolution:** 0.65 x 0.65 x 2.5 µm (xyz)  
- **ROI size:** 2048 x 2060 (effective 2048 x 2048)  
- **Bit depth:** 16-bit grayscale images  
- **Channels:** Neurons (anti-NeuN), Microglia (anti-Iba1), Cell nuclei (BOBO-1)  
- **Data volume:** ~15 TB per whole brain (3 channels)  
- **Imaging protocol:** Similar to Matsumoto K, et al., *Nature Protocols* 2019. Z-stacks are captured from both dorsal and ventral directions with sufficient overlap. Stitched volumes ensure accurate spatial registration.

## Available Atlas Data

- **Neuron Atlas cell data** (xyz coordinates, cell type annotations, atlas annotation IDs)  
  `"/Neurology/Neuron_Atlas/SCA_Cellome_data_original.bin"`

- **Averaged template images** for cell density (cell nuclei or neurons), compatible with the Allen Brain Atlas (10 µm scale)  
  `"/Neurology/Neuron_Atlas/iso_50um_R_ver4_nuclear_with_ventricle.tif"` or `"/Neurology/Neuron_Atlas/iso_50um_R_ver4.tif"`

- **pGM/pWM masks** (practical gray matter and practical white matter, defined by neuron ratio at 10 µm scale)  
  `"/Neurology/pGM_and_pWM_masks/"`  

**Download link:**  
[Google Drive](https://drive.google.com/drive/folders/1XrRgaWScrQQk3uV722mXu4JfQgIKu4IZ)

## Available Source Data

Analyzed data can be found under:  
`"/Neurology/Analyzed_cell_distribution_data/"`  

Folders named by condition (e.g., `APP_1m`, `APP_3m`) contain data such as `#4_APPmodel_Abeta_APP1m_1` (for Aβ data) or `#4_APPmodel_APP1m_1_2022_1102_1304` (for cell-type data). These include results after Step 1 of the workflow. By using these data, you can reproduce Steps 2–12 with the code in this repository. If you need to rerun pdfCluster-Based Cell Type Classification (Step 1-9), you can download the source data again.

- An example folder structure for **cell-type analysis** is:  
  `"/Neurology/Analyzed_cell_distribution_data/WT_1m/#4_APPmodel_Ctr1m_1_2022_1104_1550/"`

  Raw image data (e.g., `ex488_em620_FW`, `ex488_em620_RV`) keep the same folder hierarchy, though most files are placeholders. However, under `"/172000/172000_176260/"`, there are some `.bin` files (`200000.bin–200500.bin`) to demonstrate the reference structure.

- An example folder structure for **Aβ analysis** is:  
  `"/Neurology/Analyzed_cell_distribution_data/APP_9m/#4_APPmodel_Abeta_APP9m_1/"`

**Download link:**  
[Google Drive](https://drive.google.com/drive/folders/1XrRgaWScrQQk3uV722mXu4JfQgIKu4IZ)

## Overview

The primary goal of this code is to identify and analyze all cells expressing neuron-marker (anti-NeuN) and microglia-marker (anti-Iba1) across the entire brain. We integrate GPU-based cell detection and CPU-based analyses. Advanced computational methods enable:

- Single-cell and regional-level data integration  
- Registration to a Neuron Atlas  
- Advanced spatial statistical analyses  

## Basic Workflow

### Prerequisites

Before running the code, prepare a parameter file (e.g., in the `/param/` directory). Specify image size, resolution, and file paths in this parameter file.

### Steps

1. **Cell Digitization**  
   - **(1-1) GPU-based cell candidate segmentation (Dorsal/FW side)**  
     Based on a modified HDoG filter approach from Matsumoto K, et al., *Nature Protocols* 2019 ([CUBIC-informatics](https://github.com/lsb-riken/CUBIC-informatics)). Includes min-max filtering for normalization in neuron and microglia channels.  
     **Example command:**
     ```bash
     docker compose run dev python script/HDoG_gpu.py \
       param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_HDoG_FW.json \
       --exec build/src/3D/HDoG3D_NeuN_ver3_Rank_simple_3_color
     ```

   - **(1-2) GPU-based cell candidate segmentation (Ventral/RV side)**  
     Similar to (1-1), but for ventral-side Z-stack imaging.  
     **Example command:**
     ```bash
     docker compose run dev python script/HDoG_gpu.py \
       param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_HDoG_FW.json \
       --exec build/src/3D/HDoG3D_NeuN_ver3_Rank_simple_3_color
     ```

   - **(1-3) Merge Local to Global Coordinates**  
     Roughly transforms local stack coordinates to a global whole-brain coordinate system (pre-stitching).  
     **Example command:**
     ```bash
     python script/MergeBrain_NeuN.py full \
       param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_merge.json
     ```

   - **(1-4) Cell Nuclei Classification**  
     Classifies cell nuclei based on normalized intensity (min-max filter) and structureness.  
     **Example command:**
     ```bash
     python script/HDoG_classifier_NeuN.py \
       param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_classify.json
     ```

   - **(1-5) Image and Cell Point Stitching**  
     Uses template matching to determine stitching parameters and applies them to cell coordinates. Finalizes 3D spatial positions of all cell points.  
     **Notebook:** `/script/stitching_2023/1-5_Robust_stitching_test.ipynb`

   - **(1-6) Creating an 80 µm Voxel Cell Density Image**  
     Generates an 80 µm voxel-resolution cell density image for registration.  
     **Notebook:** `/script/1_6_Stitched_80um_image_making.ipynb`

   - **(1-7) Whole-Brain Registration to Raw Neuron Atlas Space**  
     Registers the cleared brain (CUBIC-L/CUBIC-R+) to the Allen Brain Atlas.  
     **Example command:**
     ```bash
     python script/AtlasMapping_stitched_initial_annotation_all.py annotation \
       param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_mapping_R.json -p 20
     ```

   - **(1-8) Rank Filter Normalization**  
     **Example command:**
     ```bash
     python script/MultiChannelVerification-rank-simple-dsb.py \
       param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_multichannel-rank.json -p 5
     ```

   - **(1-9) pdfCluster-Based Cell Type Classification**  
     **Notebook:** `/script/1-9_pdfCluster.ipynb`

2. **Cell Points Registration and Anatomical Annotation**  
   Assigns anatomical region annotations compatible with the Allen Brain Atlas space.  
   **Notebook:** `/script/2_SCA_registration_and_annotation.ipynb`

   **For Aβ data:**  
   **Notebook:** `/script/2_Abeta_data_registration_to_SCA.ipynb`

## Advanced Analyses

3. **Calculation of local cell density consistency after registration**  
4. **Whole-brain regional plots**  
5. **Age-dependent linear regression analysis**  
6. **Onset analysis**  
7. **Pathological pseudotime analysis**  
8. **Thal phase analysis**  
9. **Spatial density variation analysis and linear regression for biomarker 2D recall**  
10. **Microglial 3D morphology and microgliosis analysis**  
11. **Microglial analysis using pGM/pWM masks and intermediate segmentations**  
12. **Spatial single-cell risk analysis**

## Summary of Results

1. **Statistical Analysis Summary (Figs and Extended Data)**  
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
