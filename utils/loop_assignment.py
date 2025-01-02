def assign_label(label,start1,start2,max_row_size,max_col_size,label_class=1,radius=3):
    """
    label: a 2D array with the same size of submatrix, all zeros
    start1: a list of start1, indicating relative loop x coordinates in the submatrix
    start2: a list of start2, indicating relative loop y coordinates in the submatrix
    max_range: the maximum range of the submatrix,
    label_class: the label class to assign, 1 indicates loop region
    radius: the radius of the loop to assign, default is 3

    """
    for i in range(len(start1)):
        cur_start1 = max(0,start1[i]-radius)
        cur_end1 = min(max_row_size,start1[i]+radius)

        cur_start2 = max(0,start2[i]-radius)
        cur_end2 = min(max_col_size,start2[i]+radius)
        #convert to integer
        cur_start1 = int(cur_start1)
        cur_end1 = int(cur_end1)
        cur_start2 = int(cur_start2)
        cur_end2 = int(cur_end2)

        cur_start1 +=1
        cur_start2 +=1#change overall diameter to 5

        label[cur_start1:cur_end1,cur_start2:cur_end2] = label_class
        label[cur_start2:cur_end2,cur_start1:cur_end1] = label_class
    return label_class