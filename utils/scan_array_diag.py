import numpy as np
import pickle
from scipy.sparse import coo_matrix

import os
def write_pickle(output_dict,output_path):
    """
    output_dict: dict, output dictionary
    output_path: str, output path
    """
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)

def scan_matrix(matrix, input_row_size,input_col_size, stride,
                hic_count,output_dir,current_chrom):
    """
    matrix: 2D array
    input_row_size: int, row size of scanned output submatrix
    input_col_size: int, column size of scanned output submatrix
    stride: int, row stride
    hic_count: int, total read count of the Hi-C experiments
    output_dir: str, output directory
    current_chrom: str, current chromosome
    """
    row_size = matrix.shape[0]
    col_size = matrix.shape[1]
    count_save=0
    region_size = input_row_size * input_col_size
    for i in range(0, row_size - input_row_size//2, stride):
        j = i
        submatrix = np.zeros((input_row_size, input_col_size))
        row_start = max(0,i)
        row_end = min(row_size, i + input_row_size)
        col_start = max(0,j)
        col_end = min(col_size, j + input_col_size)
        submatrix[:row_end-row_start,:col_end-col_start] = matrix[row_start: row_end, col_start: col_end]
        #filter out the submatrices with too many zeros
        count_useful = np.count_nonzero(submatrix)
        if count_useful < 1:
            continue
        
        output_dict={}
        output_dict['input']=submatrix
        output_dict['input_count']=hic_count
        # modify the target according to your task
        """
        "2d_target": the output Hi-C/scHi-C matrix in scipy.sparse or numpy.array format, shape: (M,N); (optional)
        "embed_target": the embedding 1D vector in numpy.array format, shape: (512);  (optional)
        "1d_target": the 1D target vector in numpy.array format, shape: (M); (optional)
        """
        output_path = os.path.join(output_dir, str(current_chrom) + '_' + str(i) + '_' + str(j) + '.pkl')
        write_pickle(output_dict,output_path)
        count_save+=1
        if count_save%100==0:
            print('Processed %d submatrices' % count_save, " for chromosome ", current_chrom)
        
    return 

def scan_pickle(input_pkl_path, input_row_size,input_col_size, 
                stride,output_dir):
    """
    input_pkl_path: str, input pickle path  
    input_row_size: int, row size of scanned output submatrix
    input_col_size: int, column size of scanned output submatrix
    stride: int, row stride
    output_dir: str, output directory
    """

    os.makedirs(output_dir, exist_ok=True)

    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)
    total_count = 0
    for key in data:
        matrix = data[key]
        if isinstance(matrix, np.ndarray):
            cur_count = np.sum(matrix)
        elif isinstance(matrix, coo_matrix):
            cur_count = matrix.sum()
        else:
            print("Type not supported", type(matrix))
            exit()
        total_count += cur_count
    print("Total read count of Hi-C: ", total_count)        

    for key in data:
        matrix = data[key]
        if isinstance(matrix, coo_matrix):
            matrix = matrix.toarray()
            #get the symmetrical one 
            upper_tri = np.triu(matrix,1)
            all_triu = np.triu(matrix)
            matrix = all_triu + upper_tri.T
        current_chrom = str(key)
        if "chr" not in current_chrom:
            current_chrom = "chr" + current_chrom

        scan_matrix(matrix, input_row_size,input_col_size, stride,
                    total_count,output_dir,current_chrom)
"""
This script is to generate the submatrices from the Hi-C matrix.
```
python scan_array_diag.py --input_pkl_path [pkl_path] --input_row_size [submat_row_size] 
    --input_col_size [submat_col_size] --stride [stride]  --output_dir [output_dir] 
```
- input_pkl_path: str, input pickle path
- input_row_size: int, row size of scanned output submatrix
- input_col_size: int, column size of scanned output submatrix
- stride: int, row stride to scan across the matrix
- output_dir: str, output directory
"""
#run with the simple command line
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pkl_path', type=str, required=True)
    parser.add_argument('--input_row_size', type=int, required=True)
    parser.add_argument('--input_col_size', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    input_pkl_path = os.path.abspath(args.input_pkl_path)
    output_dir = os.path.abspath(args.output_dir)
    scan_pickle(input_pkl_path, args.input_row_size, args.input_col_size, 
                args.stride, output_dir)



            