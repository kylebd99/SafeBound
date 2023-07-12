import pandas as pd
import numpy as np
import sys,os
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory + 'Source/ExperimentUtils')
from InferenceUtils import *

if __name__ == '__main__':
    
    #benchmarks = ['Stats']
    benchmarks = ['JOBLight', 'JOBLightRanges', 'JOBM', 'Stats']

    PostgresParams = [10, 100, 1000, 5000, 10000]

    Postgres2DParams = [10, 100, 1000, 5000, 10000]
    
    
    for i in range(1,6):
        for benchmark in benchmarks:
            statsFile = rootFileDirectory + "StatObjects/SafeBound_" + str(i)  + "_" + benchmark + ".pkl"
            outputFile = rootFileDirectory + "Data/Results/SafeBound_Inference_" + str(i) + "_" + benchmark + ".csv"
            evaluate_inference(method = 'SafeBound', 
                               statsFile =  statsFile,
                               benchmark = benchmark,
                               outputFile = outputFile,
                               statisticsTarget = None)
            
    for i in range(1,6):
        for benchmark in benchmarks:
            outputFile = rootFileDirectory + "Data/Results/Postgres_Inference_" + str(i) + "_" + benchmark  + ".csv"
            evaluate_inference(method = 'Postgres', 
                               statsFile =  None,
                               benchmark = benchmark,
                               outputFile = outputFile,
                               statisticsTarget = PostgresParams[i-1])
     