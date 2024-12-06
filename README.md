# Neuron Atlas analysis
This repository contains code for the whole-brain-scale analysis of Cell-type (Neuron, microglia, and cell nuclei) detection, utilizing datasets imaged through tissue clearing and 10x light-sheet microscopy (xy FWHM resolution: 1.19 um, z FWHM resolution: 3.39 um) and Neuon Atlasへsingle-cell levelでのregistration、regional and single cell levelでの細胞分布の統計解析、spatial transcriptomeとの統合を含むsingle-cell risk解析を含みます. データはすべてxyzが0.65x0.65x2.5 umのvoxel resolution, ROI sizeは2048x2060 (2048x2048のみが有効なROI範囲),16bit gray scaleの画像, Neuron, microglia, and cell nucleiの3チャネルで撮影されたマウス全脳データ(1脳あたりtotal 約15TB)を想定しています。マウス脳は、previous study (Matsumoto K, et al., Nature Protocols 2019)とほぼ同様の撮影方式を撮影しており、dorsal側のxyのTileに対してZ stackをとることを繰り返して、全xy tileを撮影し、180度回転させて、ventral側のxy tileのz stackを撮影されました。xyz, 回転方向も、いずれも十分なoverlapをもつように撮影され、stitcingにより厳密な空間位置が補正されました。
Neuron Atlasの細胞データ(xyz座標, cell type(Neuron or not),atlas annntation ID)、画像registerのreferenceとなる細胞密度のaverage template image (cell nuclear or neuron, Allen Brain Atlasの10um scaleのtemplateに互換性あり) はこのリンクでダウンロードできます: hogehoge (後日update予定)
10um VoxelあたりのNeuron ratioをもとに決めたpractical gray matter (pGM), practical white matter (pWM)のregion annotation、およびそれらをtransistionalに分割したmask画像データ(Allen Brain Atlasの10um scaleのtemplateに互換性あり)はこのリンクでダウンロードできます: hogehoge (後日update予定)
B6JのWild type(1,3,5,7,9か月例, オス), APPNL-G-Fモデル(1,3,5,7,9か月例, オス), VCPモデル(8-9 weeks old, オス)、TMTモデル(8 weeks old, オス)の細胞データ(xyz座標, cell type(Neuronやmicrogliaであるかどうか),atlas annntation ID)、registerに使用したtemplate画像に関しては、このリンクでダウンロードできます: hogehoge (後日update予定)


## Overview
The primary objective of this code is to facilitate the identification and analysis of all cells with neuron-marker (anti-NeuN) and microglia-marker (anti-Iba1) posiveかどうか across the entire brain. Our methods integrate several advanced computational techniques and visualization tools to achieve this goal. Cell detectionの部分はGPUを用いた解析を、それ以外はCPUベースの解析を遂行します。




## Basic Workflow
コードを実行する前に、/param/を例としたようなparameter fileを作製する必要があります。そこへ画像のサイズや解像度、保存されているpathの場所などをinputする必要があります。
The basic procedure for analyzing whole-brain cell-type detection consists of the following steps:

1. Cell digitization
  1-1. GPU-based cell candidate segmentation of FW方向: これはprevious study (Matsumoto K, et al., Nature Protocols 2019, github: https://github.com/lsb-riken/CUBIC-informatics)を改変したもので、主に、Neuron, microgliaのチャネルにおけるmin-max filterによるnormalize値の取得する点が追加されました。FW方向というのは、脳のdorsal側のことで、脳の深部から順に表面へdorsal側に順番にZ stack撮影されたデータを基にしています。commandラインで実行します。
   　　example command: docker compose run dev python script/HDoG_gpu.py param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_HDoG_FW.json --exec build/src/3D/HDoG3D_NeuN_ver3_Rank_simple_3_color
  1-2. GPU-based cell candidate segmentation of RV方向: これはprevious study (Matsumoto K, et al., Nature Protocols 2019, github: https://github.com/lsb-riken/CUBIC-informatics)を改変したもので、主に、Neuron, microgliaのチャネルにおけるmin-max filterによるnormalize値の取得する点が追加されました。FW方向というのは、脳のventral側のことで、脳の深部から順に脳の表面へventral側へ順番にZ stack撮影されたデータを基にしています。commandラインで実行します。
   　　example command: docker compose run dev python script/HDoG_gpu.py param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_HDoG_FW.json --exec build/src/3D/HDoG3D_NeuN_ver3_Rank_simple_3_color
  1-3. Merge
   example command: python script/MergeBrain_NeuN.py full param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_merge.json
  1-4. Cell nuclei classify
   example command: python script/HDoG_classifier_NeuN.py param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_classify.json;
  1-5. Stitching image and cell points
  1-6. Making stitched 80um voxel cell density image for registration source
  1-7. Whole-brain image registration and cell point transfer to raw Neuron Atlas space (Allen Brain Atlasへのreister前の空間をraw Neuron Atlasと呼んでいます。CUBIC-L/CUBIC-R+ protocolでtissue clearingされた脳はわずかに膨潤しており、そのデータはこのAtlasへwell reisterできます)
   example command: python script/AtlasMapping_stitched_initial_annotation_all.py annotation param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_mapping_R.json -p 20;
  1-8. Rank filter normalization
   example command: python script/MultiChannelVerification-rank-simple-dsb.py param/Neuronomics/#4_APPmodel_Ctr1m_1_2022_1104_1550/param_multichannel-rank.json -p 5;
  1-9. pdfCluster
   code: /script/1-9_pdfCluster.ipynb
  1-10. Cell point transfer to Neuron Atlas (Allen Brain Atlas spaceに互換性あり)
   code: /script/1-10_SCA_SCA_registration_and_annotation.ipynb
2.  





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

