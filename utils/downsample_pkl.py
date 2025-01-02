import sys
import os
from collections import defaultdict
import pickle
import numpy as np
from scipy.sparse import coo_matrix
def array_to_coo(array):
    """
    Convert a regular 2D NumPy array to a scipy.sparse.coo_matrix.

    Parameters:
    - array (numpy.ndarray): The input 2D array.

    Returns:
    - scipy.sparse.coo_matrix: The converted COO matrix.
    """
    # Find the non-zero elements in the array
    row, col = np.nonzero(array)

    # Get the values of the non-zero elements
    data = array[row, col]

    # Create the COO matrix
    coo_mat = coo_matrix((data, (row, col)), shape=array.shape)

    return coo_mat

def sparse2tag(coo_mat):
    tag_len = coo_mat.sum()
    tag_len = int(tag_len)
    tag_mat = np.zeros((tag_len, 2))
    tag_mat = tag_mat.astype(int)
    row, col, data = coo_mat.row, coo_mat.col, coo_mat.data
    start_idx = 0
    for i in range(len(row)):
        end_idx = start_idx + int(data[i])
        tag_mat[start_idx:end_idx, :] = (row[i], col[i])
        start_idx = end_idx
    return tag_mat, tag_len
def tag2sparse(tag, nsize):
    """
    Coverts a coo-based tag matrix to sparse matrix.
    """
    coo_data, data = np.unique(tag, axis=0, return_counts=True)
    row, col = coo_data[:, 0], coo_data[:, 1]
    sparse_mat = coo_matrix((data, (row, col)), shape=(nsize, nsize))
    return sparse_mat

def downsampling_sparce(matrix, down_ratio, verbose=False):
    """
    Downsampling method for sparse matrix.
    """
    if verbose: print(f"[Downsampling] Matrix shape is {matrix.shape}")
    tag_mat, tag_len = sparse2tag(matrix)
    sample_idx = np.random.choice(tag_len, int(tag_len *down_ratio))
    sample_tag = tag_mat[sample_idx]
    if verbose: print(f'[Downsampling] Sampling {down_ratio} of {tag_len} reads')
    down_mat = tag2sparse(sample_tag, matrix.shape[0])
    return down_mat


def downsample_pkl(input_pkl, output_pkl, downsample_rate):
    data = pickle.load(open(input_pkl, 'rb'))
    return_dict={}
    for chrom in data:
        current_data = data[chrom]
        if current_data.shape[0] <=100:
            continue
        #if it is numpy array convert to sparse matrix
        if isinstance(current_data, np.ndarray):
            current_data = array_to_coo(current_data)
            
        downsampled_data = downsampling_sparce(current_data, downsample_rate,verbose=1)
        return_dict[chrom] = downsampled_data
    pickle.dump(return_dict, open(output_pkl, "wb"))
    print("finish downsampling %s"%output_pkl)
"""
This script is used to downsample the input pickle file.
```
python3 downsample_pkl.py [input.pkl] [output.pkl] [downsample_rate]
```
[input.pkl]: the input pickle file. <br>
[output.pkl]: the output pickle file. <br>
[downsample_rate]: the downsample rate [float].
"""


if __name__ == '__main__':
    if len(sys.argv)!=4:
        print("Usage: python3 downsample_pkl.py [input.pkl] [output.pkl] [downsample_rate]")
        print("This script is used to downsample the input pickle file.")
        print("[input.pkl]: the input pickle file")
        print("[output.pkl]: the output pickle file")
        print("[downsample_rate]: the downsample rate [float].")
        sys.exit(1)
    input_pkl = os.path.abspath(sys.argv[1])
    output_pkl = os.path.abspath(sys.argv[2])
    output_dir = os.path.dirname(output_pkl)
    os.makedirs(output_dir, exist_ok=True)    
    downsample_rate = float(sys.argv[3])
    downsample_pkl(input_pkl, output_pkl, downsample_rate)