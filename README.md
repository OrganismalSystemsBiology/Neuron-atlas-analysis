# Neuron Atlas analysis
This repository contains code for the whole-brain-scale analysis of Cell-type (Neuron, microglia, and cell nuclei) detection, utilizing datasets imaged through tissue clearing and 10x light-sheet microscopy (xy FWHM resolution: 1.19 um, z FWHM resolution: 3.39 um). データはすべてxyzが0.65x0.65x2.5 umのvoxel resolution, ROI sizeは2048x2060 (2048x2048のみが有効なROI範囲), 3チャネルで撮影されたマウス全脳データ(1脳あたりtotal 約15TB)を想定しています。
Neuron Atlasの細胞データ(xyz座標, cell type(Neuron or not),atlas annntation ID)、細胞密度のaverage template image (cell nuclear or neuron) はこのリンクでダウンロードできます: hogehoge (後日update予定)
B6JのWild type(1,3,5,7,9か月例, オス), APPNL-G-Fモデル(1,3,5,7,9か月例, オス), VCPモデル、TMTモデルの細胞データに関しては、このリンクでダウンロードできます: hogehoge (後日update予定)


## Overview
The primary objective of this code is to facilitate the identification and analysis of all cells with neuron-marker and microglia-marker posiveかどうか across the entire brain. Our methods integrate several advanced computational techniques and visualization tools to achieve this goal. Cell detectionの部分はGPUを用いた解析を、それ以外はCPUベースの解析を遂行します。




## Basic Workflow
The basic procedure for analyzing whole-brain cell-type detection consists of the following steps:

1. Expert Review: Initially, raw data undergo a thorough review by experts to exclude any non-specifically stained anatomical regions.
2. Voxel Scale Normalization: Data are normalized to a resolution of 8.3x8.3x8.3 µm to ensure uniformity across different datasets.
3. Tau Deposition ROI Extraction: Regions of interest (ROIs) for tau deposition are identified using a Gaussian-mean difference method. This is implemented in the **Tau extraction.ipynb** notebook.
4. CUBIC-Cloud Registration: Following ROI extraction, data are registered to a mouse brain atlas using CUBIC-Cloud (https://cubic-cloud.com/), a software tool designed for whole-brain visualization and analysis. The guidelines were detailed in Mano T et al., CellRepMethods, 2021 ([DOI](https://doi.org/10.1016/j.crmeth.2021.100038)https://doi.org/10.1016/j.crmeth.2021.100038).





## Advanced Analysis
Further analysis is conducted through the following advanced procedures:

1. Tau Gradient Consensus Analysis: Detailed workflow for consensus analysis on tau gradients, structured in **Tau gradient consensus analysis.ipynb**.
2. Whole Brain Data Visualization: Comprehensive tools and methods for visualizing the whole brain pTau depositions data, structured in **Tau gradient consensus analysis.ipynb**.





## Summary of results
1. Summary of Regional pTau Depositions: Refer to Supplementary Table 1, "**Supplementary Table 1. Regional tau profile data with each anatomical annotation.xlsx**" for a detailed breakdown.
2. Source pTau Distribution Data for Three Samples: See Supplementary Table 2, "**Supplementary Table 2. Source data for single-deposition-level tau profiles from three 18-month-old Rosa26-KI Tau++tTA+ mice**" for specific data.

Note: This code does not currently include source image data from light-sheet imaging.





## System Requirements
This code has been tested on a CentOS Linux release 7.9.2009 (Core) PCでvirtualenv environmentでwithin a Python 3.6.8やPython 3.9.0がコードのavailabilityに従って選択され、適宜使用された。部分的には、 Ubuntu 22.04.4 LTSで、Python 3.7.17やPython 3.9.19が、コードのavailabilityに従って選択され、適宜使用された。





## Citation
If you utilize this code in your research, please cite our paper:
#### Whole-Brain Single-Neuron Atlas Reveals Neuronal Vulnerability Modulated by Spatial Neuron-Microglia Homeostatic Relationship
Tomoki T. Mitani, et al

DOI: to be updated

