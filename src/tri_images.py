import numpy as np
import os
import stat
from tqdm import tqdm
import concurrent.futures as ccf
import tranche


def compute_diff(i, files):
    diff_row = np.zeros(len(files))
    dsi = tranche.Tranche(files[i])
    for j in range(i + 1, len(files)):
        dsj = tranche.Tranche(files[j])
        diff_row[j] = dsi.compare(dsj)[1]
    return i, diff_row

def parallel_compute_diff(files):
    diff = np.zeros((len(files), len(files)))
    with ccf.ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_diff, i, files) for i in range(len(files))]
        for future in tqdm(ccf.as_completed(futures), total=len(futures), desc='Ensuring data coherence'):
            i, diff_row = future.result()
            diff[i] = diff_row
    return diff

def uniforme(l:list):
    a0 = l[0]
    i = 1
    ai = l[i]
    while ai==a0 and i<len(l):
        i+=1
        ai = l[i]
    if i ==  len(l) :
        return True 
    else : 
        return False

