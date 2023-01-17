import pandas as pd
import numpy as np
import os
import sys
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory + 'Source/ExperimentUtils')
from JoinSizeInferenceUtils import *

if __name__ == '__main__':
    
    benchmarks = ['JOBM', 'JOBLight', 'Stats', 'JOBLightRanges']
    
    PostgresParams = [10, 100, 1000, 5000, 10000]
    Postgres2DParams = [10, 100, 1000, 5000, 10000]
    
    PostgresRun = 2
    safeBoundRun = 4
    
    
    for benchmark in benchmarks:
        fileExt =  ".csv"
        '''
        statsFile = rootFileDirectory + "StatObjects/SafeBound_" + str(safeBoundRun)  + "_" + benchmark + ".pkl"
        outputFile = rootFileDirectory + "Data/Results/SafeBound_Join_Size_Inference_" + benchmark + fileExt
        evaluate_join_size_inference(method = 'SafeBound', 
                           statsFile =  statsFile,
                           benchmark = benchmark,
                           outputFile = outputFile,
                           statisticsTarget = None)
        
        statsFile = rootFileDirectory + "StatObjects/Simplicity_" + benchmark + ".pkl"
        outputFile = rootFileDirectory + "Data/Results/Simplicity_Join_Size_Inference_" + benchmark + fileExt
        evaluate_join_size_inference(method = 'Simplicity', 
                           statsFile =  statsFile,
                           benchmark = benchmark,
                           outputFile = outputFile,
                           statisticsTarget = None)
        
        outputFile = rootFileDirectory + "Data/Results/Postgres_Join_Size_Inference_" + benchmark + fileExt
        evaluate_join_size_inference(method = 'Postgres', 
                           statsFile =  None,
                           benchmark = benchmark,
                           outputFile = outputFile,
                           statisticsTarget = PostgresParams[PostgresRun-1])
        
        '''
        outputFile = rootFileDirectory + "Data/Results/Postgres2D_Join_Size_Inference_" + benchmark + fileExt
        evaluate_join_size_inference(method = 'Postgres2D', 
                           statsFile =  None,
                           benchmark = benchmark,
                           outputFile = outputFile,
                           statisticsTarget = PostgresParams[PostgresRun-1])
        
        '''
        outputFile = rootFileDirectory + "Data/Results/BayesCard_Join_Size_Inference_" + benchmark + fileExt
        ensembleDirectory = rootFileDirectory + "StatObjects/BayesCardEnsembles/" + benchmark +"/"
        evaluate_join_size_inference(method = 'BayesCard', 
                           statsFile =  ensembleDirectory,
                           benchmark = benchmark,
                           outputFile = outputFile)
        outputFile = rootFileDirectory + "Data/Results/PessemisticCardinality_Join_Size_Inference_" + benchmark + fileExt
        evaluate_join_size_inference(method = 'PessemisticCardinality', 
                           statsFile =  None,
                           benchmark = benchmark,
                           outputFile = outputFile)
        
        outputFile = rootFileDirectory + "Data/Results/NeuroCard_Join_Size_Inference_" + benchmark + fileExt
        evaluate_join_size_inference(method = 'NeuroCard', 
                           statsFile =  None,
                           benchmark = benchmark,
                           outputFile = outputFile)
        '''