import numpy as np
import pickle
from scripy.sparse import coo_matrix
import os
def write_pickle(output_dict,output_path):
    """
    output_dict: dict, output dictionary
    output_path: str, output path
    """
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)

def scan_matrix(matrix, input_row_size,input_col_size, stride_row,
                stride_col,hic_count,output_dir,current_chrom,
                filter_threshold=0.05):
    """
    matrix: 2D array
    input_row_size: int, row size of scanned output submatrix
    input_col_size: int, column size of scanned output submatrix
    stride_row: int, row stride
    stride_col: int, column stride
    hic_count: int, total read count of the Hi-C experiments
    output_dir: str, output directory
    current_chrom: str, current chromosome
    """
    row_size = matrix.shape[0]
    col_size = matrix.shape[1]
    count_save=0
    region_size = input_row_size * input_col_size
    for i in range(0, row_size - input_row_size//2, stride_row):
        for j in range(0, col_size - input_col_size//2, stride_col):
            submatrix = np.zeros((input_row_size, input_col_size))
            row_start = max(0,i)
            row_end = min(row_size, i + input_row_size)
            col_start = max(0,j)
            col_end = min(col_size, j + input_col_size)
            submatrix[:row_end-row_start,:col_end-col_start] = matrix[row_start: row_end, col_start: col_end]
            #filter out the submatrices with too many zeros
            count_useful = np.count_nonzero(submatrix)
            if count_useful < region_size * filter_threshold:
                continue
            
            output_dict={}
            output_dict['input']=submatrix
            output_dict['input_count']=hic_count
            #judge if the diag is possibly included
            if col_start < row_start and col_end >row_start:
                output_dict['diag']=abs (col_start-row_start)
            elif col_start == row_start:
                output_dict['diag']=0
            elif col_start> row_start and col_start < row_end:
                output_dict['diag']= -abs (col_start-row_start)
            else:
                output_dict['diag']=None
            output_path = os.path.join(output_dir, str(current_chrom) + '_' + str(i) + '_' + str(j) + '.pkl')
            write_pickle(output_dict,output_path)
            count_save+=1
            if count_save%100==0:
                print('Processed %d submatrices' % count_save, " for chromosome ", current_chrom)
        
    return 

def scan_pickle(input_pkl_path, input_row_size,input_col_size, stride_row,
                stride_col,output_dir,filter_threshold):
    """
    input_pkl_path: str, input pickle path  
    input_row_size: int, row size of scanned output submatrix
    input_col_size: int, column size of scanned output submatrix
    stride_row: int, row stride
    stride_col: int, column stride
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

        scan_matrix(matrix, input_row_size,input_col_size, stride_row,
                stride_col,total_count,output_dir,current_chrom,filter_threshold)
"""
This script is to generate the submatrices from the Hi-C matrix.
```
python scan_array.py --input_pkl_path [pkl_path] --input_row_size [submat_row_size] 
    --input_col_size [submat_col_size] --stride_row [stride_row] 
    --stride_col [stride_col] --output_dir [output_dir] 
    --filter_threshold [filter_threshold]
```
- input_pkl_path: str, input pickle path
- input_row_size: int, row size of scanned output submatrix
- input_col_size: int, column size of scanned output submatrix
- stride_row: int, row stride to scan across the matrix
- stride_col: int, column stride to scan across the matrix
- output_dir: str, output directory
- filter_threshold: float, the threshold to filter out the submatrices with too many zeros
"""
#run with the simple command line
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pkl_path', type=str, required=True)
    parser.add_argument('--input_row_size', type=int, required=True)
    parser.add_argument('--input_col_size', type=int, required=True)
    parser.add_argument('--stride_row', type=int, required=True)
    parser.add_argument('--stride_col', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--filter_threshold', type=float, default=0.05)
    args = parser.parse_args()
    input_pkl_path = os.path.abspath(args.input_pkl_path)
    output_dir = os.path.abspath(args.output_dir)
    scan_pickle(input_pkl_path, args.input_row_size, args.input_col_size, 
                args.stride_row, args.stride_col, output_dir, args.filter_threshold)



            