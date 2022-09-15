import pandas as pd
import numpy as np
import sys, os
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory+ 'Source/ExperimentUtils')
from TrueCardUtils import *

if __name__ == '__main__':
    
    benchmarks = ['JOBLight', 'JOBLightRanges', 'JOBM', 'Stats']
    
    for benchmark in benchmarks:
        outputFile = rootFileDirectory + "StatObjects/TrueCardinality_" + benchmark + ".pkl"
        gather_true_cardinalities(benchmark = benchmark,
                                  outputFile = outputFile,)
