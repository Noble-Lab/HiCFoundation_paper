import os 
import numpy as np
from ops.sparse_ops import array_to_coo
from scipy.sparse import coo_matrix
from collections import defaultdict
def write_pkl(return_dict,output_pkl_path):
    import pickle
    with open(output_pkl_path,'wb') as f:
        pickle.dump(return_dict,f)
    print("finish writing to:",output_pkl_path)
def load_pkl(input_pkl):
    import pickle
    with open(input_pkl,'rb') as f:
        return_dict = pickle.load(f)
    return return_dict

def read_text(input_file,config_resolution):
    #records should be readID chr1 pos1 chr2 pos2
    #read line by line to get the sparse matrix
    final_dict=defaultdict(list)
    with open(input_file,'r') as f:
        for line in f:
            line = line.strip().split()
            try:
                chr1 = line[1]
                chr2 = line[3]
                pos1 = int(line[2])//config_resolution
                pos2 = int(line[4])//config_resolution
                final_dict[(chr1,chr2)].append((pos1,pos2))
            except:
                print("*"*40)
                print("Skip line in records:",line)
                print("The line should be in format of [readID chr1 pos1 chr2 pos2]")
                print("*"*40)
    return final_dict

def countlist2coo(input_dict):
    final_dict={}
    for key in input_dict:
        row=[]
        col=[]
        data=[]
        for item in input_dict[key]:
            row.append(item[0])
            col.append(item[1])
            data.append(1)
        max_size = max(max(row),max(col))+1
        cur_array = coo_matrix((data,(row,col)),shape=(max_size,max_size))
        #sum duplicates
        cur_array.sum_duplicates()
        final_dict[key]=cur_array
    return final_dict
def convert_to_pkl(input_file, output_pkl,config_resolution):
    
    output_dir = os.path.dirname(output_pkl)
    os.makedirs(output_dir,exist_ok=True)
    #convert to .pkl format
    initial_dict = read_text(input_file,config_resolution)
    #filter intra-chromosome regions
    final_dict = {}
    for key in initial_dict:
        if key[0] == key[1]:
            final_dict[key[0]] = initial_dict[key]
    #then change it to coo_matrix array
    return_dict = countlist2coo(final_dict)
    write_pkl(return_dict,output_pkl)
"""
This script is used to convert .pairs format to .pkl format
```
python pairs2array.py [input_file] [output_pkl] [resolution]
```
[input_file]: the input file in .pairs format
[output_pkl]: the output file in .pkl format
[resolution]: the resolution of the input Hi-C data
"""
import sys
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python pairs2array.py [input_file] [output_pkl] [resolution]")
        print("[input_file]: the input file in .pairs format")
        print("[output_pkl]: the output file in .pkl format")
        print("[resolution]: the resolution of the input Hi-C data")
        exit()
    input_file = sys.argv[1]
    output_pkl = sys.argv[2]
    config_resolution = int(sys.argv[3])
    input_file = os.path.abspath(input_file)
    output_pkl = os.path.abspath(output_pkl)

    convert_to_pkl(input_file,output_pkl,config_resolution)


