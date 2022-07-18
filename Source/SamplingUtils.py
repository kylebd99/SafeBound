import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import murmurhash as mmh
import math 
import queue

def generate_zipf(z, size):
    pmf = []
    s = 0.
    for i in range(size):
        val = 1/(i+1)**z
        s += val
        pmf.append(val)
    for i in range(size):
        pmf[i] /= s
    return pmf

def get_sample(table, sample_proportion, columns, seeds):
    if(len(columns) != len(seeds)):
        raise NameError('Must have equal column and seed numbers') 
    if(len(columns) == 0):
        sample = table.sample(frac=sample_proportion, random_state=np.random.randint(0, 100000))
        return sample
    for i in range(len(columns)):
        table.loc[:,"hash_" + columns[i]] = [mmh.hash(str(x), seeds[i]) % 10000 for x in table[columns[i]]]
    sample_positions = pd.Series(np.zeros(len(table), dtype=np.bool))
    for col in columns:
        sample_positions = sample_positions | (table["hash_" + col] < 10000 * sample_proportion)
    return table[sample_positions]

def calc_std_dev(vals, mean=0, calculate_mean=True):
    if(calculate_mean):
        mean = 0
        for val in vals:
            mean+= val
        mean/= len(vals)
    std_dev = 0
    for val in vals:
        std_dev += (mean - val)**2
    std_dev /= len(vals)
    std_dev = math.sqrt(std_dev)
    return std_dev
