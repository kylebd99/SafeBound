import pandas as pd
import sys, os
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory + 'Source/ExperimentUtils')
from BuildUtils import *

if __name__ == '__main__':
    
    
    benchmarks = ['JOBLight', 'JOBLightRanges', 'JOBM', 'Stats']
    
    SafeBoundParams = {benchmark : {'relativeErrorPerSegment' : [.1, .05, .02, .01, .001],
                      'numHistogramBuckets' : [8, 16, 32, 64, 128],
                      'numEqualityOutliers' : [64, 128, 512, 1028, 2056],
                      'numCDFGroups' : [4, 6, 16, 32, 64],
                      'trackNulls' : [False for _ in range(5)],
                      'trackBiGrams' : [False for _ in range(5)],
                      'numCores' : [6 for _ in range(5)],
                      'groupingMethod' : ["CompleteClustering" for _ in range(5)],
                      'modelCDF' : [True for _ in range(5)],
                      'verbose' : [False for _ in range(5)]} for benchmark in benchmarks
                      }
    SafeBoundParams["JOBM"]['numEqualityOutliers'] = [5*x for x in SafeBoundParams["JOBM"]['numEqualityOutliers']]
    SafeBoundParams["JOBM"]['trackBiGrams'] = [True for _ in range(5)]
    SafeBoundParams["JOBM"]['trackNulls'] = [True for _ in range(5)]
    SafeBoundParams["JOBM"]['verbose'] = [True for _ in range(5)]

    SafeBoundFileNames = [rootFileDirectory + "StatObjects/SafeBound2_" + str(i) for i in range(1,6)]

    PostgresParams = {'statisticsTarget' : [10, 100, 1000, 5000, 10000]}

    Postgres2DParams = {'statisticsTarget' : [10, 100, 1000, 5000, 10000]}

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
    safeBoundResults.to_csv(rootFileDirectory + "Data/Results/SafeBound2_Build_Results.csv")
    
    
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
            
    bayesCardBuildTime = []
    bayesCardSize = []
    bayesCardBenchmarks = []
    for benchmark in benchmarks:
        time, size = build_stats_object(method='BayesCard',
                           benchmark=benchmark,
                           parameters = {},
                           outputFile = rootFileDirectory + "StatObjects/BayesCardEnsembles/" + benchmark
                  )
        bayesCardBuildTime.append(time)
        bayesCardSize.append(size)
        bayesCardBenchmarks.append(benchmark)
    bayesCardResults = pd.DataFrame()
    bayesCardResults['BuildTime'] = bayesCardBuildTime
    bayesCardResults['Size'] = bayesCardSize
    bayesCardResults['Benchmark'] = bayesCardBenchmarks
    bayesCardResults['Run'] = 1
    bayesCardResults.to_csv(rootFileDirectory + "Data/Results/BayesCard_Build_Results.csv")
    
    '''
    postgres2DBuildTime = []
    postgres2DSize = []
    postgres2DBenchmarks = []
    postgres2DRuns = []
    for benchmark in benchmarks:
        for i in range(5):
            time, size = build_stats_object(method='Postgres2D',
                               benchmark=benchmark,
                               parameters = {x:y[i] for x,y in Postgres2DParams.items()},
                               outputFile = None
                      )
            postgres2DBuildTime.append(time)
            postgres2DSize.append(size)
            postgres2DBenchmarks.append(benchmark)
            postgres2DRuns.append(i+1)
    postgres2DResults = pd.DataFrame()
    postgres2DResults['BuildTime'] = postgres2DBuildTime
    postgres2DResults['Size'] = postgres2DSize
    postgres2DResults['Benchmark'] = postgres2DBenchmarks
    postgres2DResults['Run'] = postgres2DRuns
    postgres2DResults.to_csv(rootFileDirectory + "Data/Results/Postgres2D_Build_Results.csv")
    
    
    '''
            
            
            
    
        
    
    
