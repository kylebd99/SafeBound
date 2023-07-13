import pandas as pd
import sys, os
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory + 'Source/ExperimentUtils')
from BuildUtils import *

if __name__ == '__main__':
    
    
#    benchmarks = ['Stats']
    benchmarks = ['Stats', 'JOBLight', 'JOBLightRanges', 'JOBM']
    
    SafeBoundParams = {benchmark : {'relativeErrorPerSegment' : [.1, .05, .02, .01, .001],
                      'numHistogramBuckets' : [8, 16, 32, 64, 128],
                      'numEqualityOutliers' : [64, 128, 512, 1028, 2056],
                      'numCDFGroups' : [4, 6, 16, 32, 64],
                      'trackNulls' : [False for _ in range(5)],
                      'trackTriGrams' : [False for _ in range(5)],
                      'numCores' : [18 for _ in range(5)],
                      'groupingMethod' : ["CompleteClustering" for _ in range(5)],
                      'modelCDF' : [True for _ in range(5)],
                      'verbose' : [False for _ in range(5)]} for benchmark in benchmarks
                      }
    if "JOBM" in benchmarks:
        SafeBoundParams["JOBM"]['numEqualityOutliers'] = [5*x for x in SafeBoundParams["JOBM"]['numEqualityOutliers']]
        SafeBoundParams["JOBM"]['trackTriGrams'] = [True for _ in range(5)]
        SafeBoundParams["JOBM"]['trackNulls'] = [True for _ in range(5)]
        SafeBoundParams["JOBM"]['numCores'] = [6 for _ in range(5)]
        SafeBoundParams["JOBM"]['verbose'] = [True for _ in range(5)]
    
    SafeBoundFileNames = [rootFileDirectory + "StatObjects/SafeBound_" + str(i) for i in range(1,6)]
    
    safeBoundBuildTime = []
    safeBoundSize = []
    safeBoundBenchmarks = []
    safeBoundRuns = []
    for benchmark in benchmarks:
        for i in range(5):
                time, size = build_stats_object(method='SafeBound',
                                   benchmark=benchmark,
                                   parameters = {x:y[i] for x,y in SafeBoundParams[benchmark].items()},
                                   outputFile = SafeBoundFileNames[i] + "_" + benchmark + ".pkl"
                            )
                safeBoundBuildTime.append(time)
                safeBoundSize.append(size)
                safeBoundBenchmarks.append(benchmark)
                safeBoundRuns.append(i+1)
    safeBoundResults = pd.DataFrame()
    safeBoundResults['BuildTime'] = safeBoundBuildTime
    safeBoundResults['Size'] = safeBoundSize
    safeBoundResults['Benchmark'] = safeBoundBenchmarks
    safeBoundResults['Run'] = safeBoundRuns
    safeBoundResults.to_csv(rootFileDirectory + "Data/Results/SafeBound_Build_Results.csv")

    PostgresParams = {'statisticsTarget' : [10, 100, 1000, 5000, 10000]}
    Postgres2DParams = {'statisticsTarget' : [10, 100, 1000, 5000, 10000]}
    
    postgresBuildTime = []
    postgresSize = []
    postgresBenchmarks = []
    postgresRuns = []
    for benchmark in benchmarks:
        for i in range(5):
            time, size = build_stats_object(method='Postgres',
                               benchmark=benchmark,
                               parameters = {x:y[i] for x,y in PostgresParams.items()},
                               outputFile = None
                      )
            postgresBuildTime.append(time)
            postgresSize.append(size)
            postgresBenchmarks.append(benchmark)
            postgresRuns.append(i+1)
    postgresResults = pd.DataFrame()
    postgresResults['BuildTime'] = postgresBuildTime
    postgresResults['Size'] = postgresSize
    postgresResults['Benchmark'] = postgresBenchmarks
    postgresResults['Run'] = postgresRuns
    postgresResults.to_csv(rootFileDirectory + "Data/Results/Postgres_Build_Results.csv")