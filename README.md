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
### 1. All Hi-C experiments downloading
Please follow the instructions in [notebook](notebooks/pretrain_data.ipynb) to download the .hic data needed for pre-training or other purposes. <br>
To use such data for pre-training like HiCFoundation, please see [Pre-training section](#Pre-training pipeline of HiCFoundation) to use the downloaded data for pre-training purposes.




## Pre-training pipeline of HiCFoundation

<details>
<summary>Pre-training pipeline of HiCFoundation</summary>

### 1. Convert different formats to pickle array file
We can support the Hi-C experiments recorded in the following format. Please use the following script under ``utils`` directory to convert them into .pkl file for further processing. 
- .hic file: Please use [hic2array.py](utils/hic2array.py) script to convert all cis, trans contact to .pkl file.
- .cool file: Please use [cool2array.py](utils/cool2array.py) script to convert all cis, trans contact to .pkl file.
- .pairs file: Please user [pairs2array.py](utils/pairs2array.py) script to convert all contact to .pkl file.

All the instructions in run is included in the script. You can simply run the following command to get instructions for each converting script:
```
python3 [script.py]
```
Then you can see detailed instructions in the command line. 

### 2. Generate submatrix from .pkl file
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

### 3. Pre-training of HiCFoundation
After preparing the data, please follow the pre-training framework instructions on [HiCFoundation](https://github.com/Noble-Lab/HiCFoundation).  <br>
Then you can train HiCFoundation from scratch.


</details>

## Fine-tuning pipeline of HiCFoundation


## Figure visualization in HiCFoundation paper
