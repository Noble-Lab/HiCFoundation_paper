import sys
import os
import pandas as pd
import numpy as np

def extract_loc(pred_detect_path):
    info=pd.read_csv(pred_detect_path, delimiter="\t",skiprows=[1])
    #get all chromsome names
    overall_dict={}
    chroms=info['#chr1'].unique().tolist()
    for chrom in chroms:
        info_chrom=info.loc[info['#chr1']==chrom]
        info_chrom=info_chrom.loc[info_chrom['chr2']==chrom]
        x_list=info_chrom['x1'].tolist()
        y_list=info_chrom['y1'].tolist()
        coord=np.stack([x_list,y_list],axis=1)
        overall_dict[chrom]=coord
    return overall_dict
def count_all_peak(input_dict):
    count=0
    for key in input_dict:
        count+=len(input_dict[key])
    return count
def merge_loop_loc(query_loc, key_loc,max_scope=5):
    """
    query_loc: nd array N*3, query loc
    key_loc: nd array M*3, key loc
    max_scope: max distance to match
    """
    common_loc = []
    visit_label = np.zeros(len(query_loc))
    #cdist calculation
    from scipy.spatial.distance import cdist
    dist_array = cdist(key_loc,query_loc)
    for i, key_peak  in enumerate(key_loc):
        cur_dist = dist_array[i]
        cur_closest_query = int(np.argmin(cur_dist))
        if cur_dist[cur_closest_query]>=max_scope:
            continue
        #then check if the closes query's closest query is also the current key
        cur_query_closest_key = int(np.argmin(dist_array[:,cur_closest_query]))
        if i!=cur_query_closest_key:
            continue
        matched_query_peak = query_loc[cur_closest_query]
        merged_peak = np.stack([key_peak,matched_query_peak],axis=0)
        peak_center = np.mean(merged_peak,axis=0)
        common_loc.append(peak_center)
    return common_loc

#according to the consistency list
input_bed_path1 = os.path.abspath(sys.argv[1])
input_bed_path2 = os.path.abspath(sys.argv[2])
resolution = int(sys.argv[3])
output_bed_path = os.path.abspath(sys.argv[4])  
output_dir = os.path.dirname(output_bed_path)
os.makedirs(output_dir,exist_ok=True)

#use the less detected one to gen new labels
query_loc = extract_loc(input_bed_path1)
key_loc = extract_loc(input_bed_path2)
if count_all_peak(query_loc)<=count_all_peak(key_loc):
    key_loc,query_loc = query_loc,key_loc
final_loc = {}
for key in key_loc:
    if key not in query_loc:
        continue
    if "Y" in key or "M" in key or "Un" in key or "random" in key:
        continue
    common_loc = merge_loop_loc(query_loc=query_loc[key],key_loc=key_loc[key],max_scope=5*resolution)
    final_loc[key]=common_loc
#write this to output dir 

with open(output_bed_path,'w') as wfile:
    wfile.write("chr1\tx1\tx2\tchr2\ty1\ty2\n")
    for key in final_loc:
        for loc in final_loc[key]:
            wfile.write(key+"\t"+str(loc[0])+"\t"+str(loc[0]+resolution)+"\t"+
                        key+"\t"+str(loc[1])+"\t"+str(loc[1]+resolution)+"\n")

    
    