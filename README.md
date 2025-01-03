# HiCFoundation_paper

This repo includes data processing pipeline of HiCFoundation paper. 

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
Then you can see detailed instructions in the command line. <br>
We used 5kb resolution for pre-training to include more data for training.

### 3. Generate submatrix from .pkl file
Please run the following command to generate submatrices from ,pkl file:
```
python3 utils/scan_array.py --input_pkl_path [pkl_path] --input_row_size 448 \
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

<details>
<summary>Fine-tuning pipeline of HiCFoundation</summary>

### 1. Download the data from database
Please follow the instructions to download data for different tasks of HiCFoundation.
- [Reproducibility task](#hi-c-experiments-collection-from-database).
- [Chromatin loop detection task](#hi-c-experiments-collection-from-database).
- [Resolution enhancement task](#hi-c-experiments-collection-from-database).
- [Epigenomic assay profiling task](#hi-c-experiments-collection-from-database).
- [Single-cell Hi-C analysis](#hi-c-experiments-collection-from-database).

They are under different sections in the [dataset collection section](#hi-c-experiments-collection-from-database). Please check corresponding section for more details.

### 2. Convert the data to submatrix for fine-tuning
The submatrices should be saved in .pkl format for HiCFoundation fine-tuning framework processing. <br>
```
"input": the input Hi-C/scHi-C matrix in scipy.sparse or numpy.array format, shape: (M,N);
"input_count": the total count of Hi-C expriment, should be a float scalar value;  (optional)
"2d_target": the output Hi-C/scHi-C matrix in scipy.sparse or numpy.array format, shape: (M,N); (optional)
"embed_target": the embedding 1D vector in numpy.array format, shape: (512);  (optional)
"1d_target": the 1D target vector in numpy.array format, shape: (M); (optional)
```


#### 2.1 Convert Hi-C files in .pkl format
First convert all .hic files to .pkl files using specified resolution.
```
python3 utils/hic2array.py {input_hic} {output_pkl} {resolution} 0 2
```
- {input_hic} is the input Hi-C file path
- {output_pkl} is the converted .pkl file path
- {Resolution} is the resolution for analysis (integer). 25000 (25kb) for the reproducibility task, 10000 (10kb) for resolution enhancement/loop detection task, 1000 (1kb) for epigenomic assay task, 1000000 (1 Mb) for single-cell analysis
- 0 indicates None normalization applied, 2 indicates saving cis-contact in scipy.sparse coo_matrix format.


#### 2.2 Generate submatrix for different tasks
For reproducibility/loop/resolution/single-cell task, please run the following command line to generate submatrices.
```
python3 utils/scan_array_diag.py --input_pkl_path [pkl_path] --input_row_size 224 \
    --input_col_size 224 --stride 20 \
    --output_dir [output_dir] 
```
This script will generate many submatrices for fine-tuning of different tasks. 
- [pkl_path]: The processed pkl file generated from last step.
- [output_dir]: The output directory 

For epigenomic assay profiling task, please run the following command line to generate submatrices.<br>
```
python3 utils/scan_array_diag_center.py --input_pkl_path [pkl_path] --input_row_size 128 \
    --input_col_size 4000 --stride 32 \
    --output_dir [output_dir] 
```
Here we need to make sure the training samples of the center of columns in the submatrices corresponds to the center of the diagonal line.


#### 2.3 Modify submatrix information with labels
##### 2.3.1 Reproducibility analysis
No further labels are needed. Based on [Supplementary Table](data/Supplementary_Table_hicfoundation.xlsx) Sup3 sheet, embeddings of any submatrix from BR should be similar, while from NR should be different. <br>
Then please integrate triplet loss in [loss_function](https://github.com/Noble-Lab/HiCFoundation/blob/main/finetune/loss.py) in [HiCFoundation](https://github.com/Noble-Lab/HiCFoundation) repo.
```
import torch.nn as nn
import torch.nn.functional as F
criterion =  nn.TripletMarginWithDistanceLoss(
         distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),margin=1.0)
loss = criterion(anchor, positive, negative)
```
where the anchor, positive, negative are the embeddings of BR1, BR2 and NR, respectively.

##### 2.3.2 Chromatin loop detection
Please first run HiCCUPs to call loops at each BR separately by taking the processed .hic file as input.
```
python3 utils/hiccups_loop.py [hicFile] [output_dir] [resolution]
```
- [hicFile] is the processed .hic file from .bam/.pairs file (illustrated in [dataset preparing section](#hi-c-experiments-collection-from-database))
- [output_dir] is the output dir to store the detected loops in .bedpe format
- [resolution] specified resolution to run, can be choice of 5000 (5kb), 10000 (10kb) and 25000 (25kb).

Then you can merge the loop calls from two BRs, and use the consensus loop to train the model. <br>
To merge the loop calls, please run the following command
```
python3 utils/merge_BRloop.py [BR1_loop.bedpe] [BR2_loop.bedpe] [resolution] [output.bedpe]
```
- [BR1_loop.bedpe] is the hiccups loop call from BR1.hic, which stored in ``merged_loops.bedpe`` in your specified directory.
- [BR2_loop.bedpe] is the hiccups loop call from BR2.hic, which stored in ``merged_loops.bedpe`` in your specified directory.
- [resolution] is the resolution of loop calls, can be choice of 5000 (5kb), 10000 (10kb) and 25000 (25kb).
- [output.bedpe] is the specified path to store the consensus loop

You can then use the [output.bedpe] to modify each submatrce's "2d_target" key in .pkl file.
In our setting, we assigned We the neighboring 5×5 pixels (50 kb×50 kb at 10kb resolution) of the loop calls as pixel-wise loop labels. <br>
You can modify the [assign_label](utils/loop_assignment.py) function to assign pixel-level loop target for model's training.

##### 2.3.3 Resolution enhancement detection (bulk/single-cell)
Here the input should be downsampled submatrix, the output should be the original submatrix. <br>
The 1st step is to get the downsampled pair of a Hi-C experiment. You can do by the following command:
```
python3 utils/downsample_pkl.py [input.pkl] [downsample.pkl] [downsample_ratio]
```
- [input.pkl] the input pickle that includes all Hi-C information, which is processed in [dataset collection section](#hi-c-experiments-collection-from-database) by converting .hic/.cool/.pairs data.
- [downsample.pkl] the output pickle that included ddownsampled Hi-C information.
- [downsample_ratio] the downsample ratio applied to the Hi-C.

Then you can moidfy [scan_array_diag.py](utils/scan_array_diag.py) function to scan across [input.pkl] and [downsample.pkl], then the submatrices from [input.pkl] should be saved into ``2d_target`` key, and the submatrices from [downsample.pkl] should be saved into ``input`` key.


##### 2.3.4 Epigenomic assay profiling
After collecting the .bigWig files of different epigenomic assays, please first convert them into .pkl file for further processing.
```
python3 utils/bigwig2array.py [input_bw] [output_pkl] [resolution]
```
[input_bw]: the input bigwig file. <br>
[output_pkl]: the output pkl file with [chrom]:[signal] format. <br>
[resolution]: the output resolution of the signal. <br>
Here our resolution for epigenomic assay is 1000 (1kb).

After obtaining the epigenomic .pkl file, then please update "1d_target" key in the submatrix's pkl file. You can simply update the [script](utils/scan_array_diag_center.py) to also get the corresponding 1D signal from the epigenomic .pkl file and save it to the "1d_target" key in the submatrix .pkl file.


### 3. Finetune HiCFoundation using training data

Using the prepared submatrix .pkl files, please follow the instructions on [Fine-tuning framework of HiCFoundation](https://github.com/Noble-Lab/HiCFoundation/tree/main?tab=readme-ov-file#fine-tuning-hicfoundation-for-new-tasks) on HiCFoundation repo to start fine-tuning HiCFoundation for specific tasks.

</details>


## Inference pipeline of HiCFoundation

<details>
<summary>Inference pipeline of HiCFoundation</summary>
For inference of HiCFoundation for different tasks, please see instructions of [inference of HiCFoundation](https://github.com/Noble-Lab/HiCFoundation/tree/main?tab=readme-ov-file#inference-of-fine-tuned-hicfoundation) on HiCFoundation repo to do inference for different tasks. <br>
Recommended: [Google Colab](https://github.com/Noble-Lab/HiCFoundation/blob/main/HiCFoundation.ipynb). Please consider to use Google colab to do online inference if you only wanted to test a few examples, where the environment is automatically configured.

</details>

