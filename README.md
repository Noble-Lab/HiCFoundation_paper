# HiCFoundation_paper

This repo includes data processing, visualization pipeline used in HiCFoundation paper. 

HiCFoundation is a generalizable Hi-C foundation model for chromatin architecture, single-cell and multi-omics analysis across species.

Copyright (C) 2024 Xiao Wang, Yuanyuan Zhang, Suhita Ray, Anupama Jha, Tangqi Fang, Shengqi Hang, Sergei Doulatov, William Stafford Noble, and Sheng Wang

License: Apache License 2.0

Contact:  Sergei Doulatov (doulatov@uw.edu) & William Stafford Noble (wnoble@uw.edu) & Sheng Wang (swang@cs.washington.edu)

For technical problems or questions, please reach to Xiao Wang (wang3702@uw.edu) and Yuanyuan Zhang (zhang038@purdue.edu).

## Citation:
Xiao Wang, Yuanyuan Zhang, Suhita Ray, Anupama Jha, Tangqi Fang, Shengqi Hang, Sergei Doulatov, William Stafford Noble, & Sheng Wang. A generalizable Hi-C foundation model for chromatin architecture, single-cell and multi-omics analysis across species. bioRxiv, 2024. [Paper](https://www.biorxiv.org/content/10.1101/2024.12.16.628821)
<br>
```
@article{wang2024hicfoundation,   
  title={A generalizable Hi-C foundation model for chromatin architecture, single-cell and multi-omics analysis across species},   
  author={Xiao Wang, Yuanyuan Zhang, Suhita Ray, Anupama Jha, Tangqi Fang, Shengqi Hang, Sergei Doulatov, William Stafford Noble, and Sheng Wang},    
  journal={bioRxiv},    
  year={2024}    
}   
```

## HiCFoundation setup
Please follow the instructions of [HiCFoundation](https://github.com/Noble-Lab/HiCFoundation) repo to install HiCFoundation and configure its environment.


## Hi-C experiments collection from database

<details>
<summary>This section includes detailed steps for collecting data for HiCFoundation pre-training and fine-tuning. </summary>

### 1. All Hi-C experiments downloading
Please follow the instructions in [notebook](notebooks/pretrain_data.ipynb) to download the .hic data needed for pre-training or other purposes. <br>
To use such data for pre-training like HiCFoundation, please see [Pre-training section](#Pre-training-pipeline-of-HiCFoundation) to use the downloaded data for pre-training purposes.

### 2. Data for fine-tuning of reproducibility task
Please follow the instructions in [notebook](notebooks/reproducibility_data.ipynb) to download the needed files for reproducibility task. <br>
To further convert those files to .hic file for further processing, please check the 
[4DN_pipeline](https://github.com/4dn-dcic/docker-4dn-hic) to convert .bam/.pairs file to .hic file for further processing. <br>
To use such data for fine-tuning HiCFoundation, please see [Fine-tuning section](#fine-tuning-pipeline-of-hicfoundation) to use the downloaded data for fine-tuning purposes.

### 3. Data for fine-tuning of chromatin loop detection task
Please follow the instructions in [notebook](notebooks/loop_data.ipynb) to download the needed files for loop detection task. <br>
To further convert those files to .hic file for further processing, please check the 
[4DN_pipeline](https://github.com/4dn-dcic/docker-4dn-hic) to convert .bam/.pairs file to .hic file for further processing. <br>
To use such data for fine-tuning HiCFoundation, please see [Fine-tuning section](#fine-tuning-pipeline-of-hicfoundation) to use the downloaded data for fine-tuning purposes.

### 4. Data for fine-tuning of resolution enhancement task
Please follow the instructions in [notebook](notebooks/resolution_data.ipynb) to download the needed files for resolution enhancement task. <br>
To use such data for fine-tuning HiCFoundation, please see [Fine-tuning section](#fine-tuning-pipeline-of-hicfoundation) to use the downloaded data for fine-tuning purposes.

### 5. Data for fine-tuning of epigenomic assay profiling task
Please follow the instructions in [notebook](notebooks/epigenomic_data.ipynb) to download the needed files for epigenomic profiling task. <br>
To use such data for fine-tuning HiCFoundation, please see [Fine-tuning section](#fine-tuning-pipeline-of-hicfoundation) to use the downloaded data for fine-tuning purposes.

### 6. Data for fine-tuning of single-cell Hi-C analysis



### 7. Download multi-species Hi-C dataset
Please follow the instructions in [notebook](notebooks/multispecies_data.ipynb) to download the needed files for multi-species analysis. <br>
Then please run inference on the processed .pkl file following instructions in [HiCFoundation](https://github.com/Noble-Lab/HiCFoundation/tree/main#inference-of-fine-tuned-hicfoundation) repo.


### 8. Download HSPC and neutrophil data
The related Hi-C files can be downloaded from [GEO website](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174533). <br>
For example, you can download the HSPC control Hi-C files from [link](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE174533&format=file&file=GSE174533%5F1%2DC11%2DCB1%2E2%2DC11%2DCB2%2Emerge%2Ehic). <br>
Then please run inference on the .hic file following instructions in [HiCFoundation](https://github.com/Noble-Lab/HiCFoundation/tree/main#inference-of-fine-tuned-hicfoundation) repo.

</details>

## Pre-training pipeline of HiCFoundation

<details>
<summary>Pre-training pipeline of HiCFoundation</summary>

### 1. Download Hi-C data from database
Please check [1. All Hi-C experiments downloading](#1-all-hi-c-experiments-downloading) to download all Hi-C data for pre-training purposes.

### 2. Convert different formats to pickle array file
We can support the Hi-C experiments recorded in the following format. Please use the following script under ``utils`` directory to convert them into .pkl file for further processing. 
- .hic file: Please use [hic2array.py](utils/hic2array.py) script to convert all cis, trans contact to .pkl file.
- .cool file: Please use [cool2array.py](utils/cool2array.py) script to convert all cis, trans contact to .pkl file.
- .pairs file: Please user [pairs2array.py](utils/pairs2array.py) script to convert all contact to .pkl file.

All the instructions in run is included in the script. You can simply run the following command to get instructions for each converting script:
```
python3 [script.py]
```
Then you can see detailed instructions in the command line. 

### 3. Generate submatrix from .pkl file
Please run the following command to generate submatrices from ,pkl file:
```
python3 scan_array.py --input_pkl_path [pkl_path] --input_row_size 448 \
    --input_col_size 448 --stride_row 224 --stride_col 224 \
    --output_dir [output_dir] --filter_threshold 0.01
```
- pkl_path: str, input pickle path
- output_dir: str, output directory
- filter_threshold: float, the threshold to filter out the submatrices with too many zeros. Here we filtered submatrices that did not have 1% entries have nonzero reads.

The suggested submatrices output of each pkl should be put under the ``output_dir/[hic_id]``, that can be easily processed by the pre-training framework in [HiCFoundation](https://github.com/Noble-Lab/HiCFoundation) repo.

### 4. Pre-training of HiCFoundation
After preparing the data, please follow the pre-training framework instructions on [HiCFoundation](https://github.com/Noble-Lab/HiCFoundation).  <br>
Then you can train HiCFoundation from scratch.


</details>

## Fine-tuning pipeline of HiCFoundation

### 1. 


## Figure visualization in HiCFoundation paper
