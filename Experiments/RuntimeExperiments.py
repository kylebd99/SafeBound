import pandas as pd
import numpy as np
import os
import sys
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory + 'Source/ExperimentUtils')
from RuntimeUtils import *

if __name__ == '__main__':
    
    benchmarks = ['JOBLight', 'JOBLightRanges', 'JOBM', 'Stats']
    
    PostgresParams = [10, 100, 1000, 5000, 10000]
    Postgres2DParams = [10, 100, 1000, 5000, 10000]
    
    PostgresRun = 2
    safeBoundRun = 4
    
    isParrallel = False
    numberOfRuns = 5
    
    for benchmark in benchmarks:
        fileExt = None
        if isParrallel:
            fileExt = ".csv"
        else:
            fileExt = "_NP.csv"
            
        
        outputFile = rootFileDirectory + "Data/Results/TrueCardinality_Runtime_" + benchmark + fileExt
        statsFile = rootFileDirectory + "StatObjects/TrueCardinality_" + benchmark + ".pkl"
        evaluate_runtime(method = 'TrueCardinality', 
                           statsFile =  statsFile,
                           benchmark = benchmark,
                           outputFile = outputFile,
                            runs=numberOfRuns)
        
        
        statsFile = rootFileDirectory + "StatObjects/SafeBound_" + str(safeBoundRun)  + "_" + benchmark + ".pkl"
        outputFile = rootFileDirectory + "Data/Results/SafeBound_Runtime_" + str(safeBoundRun) + "_" + benchmark + fileExt
        evaluate_runtime(method = 'SafeBound', 
                           statsFile =  statsFile,
                           benchmark = benchmark,
                           outputFile = outputFile,
                           statisticsTarget = None,
                            runs=numberOfRuns)
        
            
        outputFile = rootFileDirectory + "Data/Results/Postgres_Runtime_" + str(PostgresRun) + "_" + benchmark + fileExt
        evaluate_runtime(method = 'Postgres', 
                           statsFile =  None,
                           benchmark = benchmark,
                           outputFile = outputFile,
                           statisticsTarget = PostgresParams[PostgresRun-1],
                            runs=numberOfRuns)
        
