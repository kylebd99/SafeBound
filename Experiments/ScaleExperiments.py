import pandas as pd
import sys
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory + 'Source/ExperimentUtils')
from BuildUtils import *

if __name__ == '__main__':
    
    benchmarks = ['TPCH-1', 'TPCH-3', 'TPCH-5', 'TPCH-7','TPCH-10']
    SafeBoundParams = {benchmark : {'relativeErrorPerSegment' : .01,
                      'numHistogramBuckets' : 64,
                      'numEqualityOutliers' : 1028,
                      'numCDFGroups' :  32,
                      'trackNulls' : True,
                      'trackTriGrams' : False,
                      'numCores' : 8,
                      'groupingMethod' : "CompleteClustering",
                      'modelCDF' : True,
                      'verbose' : True} for benchmark in benchmarks
                      }
    SafeBoundFileName = rootFileDirectory + "StatObjects/SafeBound_Scale_"
    
    
    buildTime = []
    statSize = []
    benchmarkDone = []
    hadTriGrams = []
    successful = []
    for benchmark in benchmarks:
        try:
            time, size = build_stats_object(method='SafeBound',
                               benchmark=benchmark,
                               parameters = {x:y for x,y in SafeBoundParams[benchmark].items()},
                               outputFile = SafeBoundFileName + benchmark + ".pkl"
                        )
            buildTime.append(time)
            statSize.append(size)
            benchmarkDone.append(benchmark)
            hadTriGrams.append(False)
            successful.append(True)
            print("BUILD FINISHED, TIME: " + str(time))
        except Exception as e:
            print(e)
            buildTime.append(0)
            statSize.append(0)
            benchmarkDone.append(benchmark)
            hadTriGrams.append(False)
            successful.append(False)
    
    safeBoundResults = pd.DataFrame()
    safeBoundResults['BuildTime'] = buildTime
    safeBoundResults['Size'] = statSize
    safeBoundResults['Benchmark'] = benchmarkDone
    safeBoundResults['HadTriGrams'] = hadTriGrams
    safeBoundResults['Successful'] = successful
    safeBoundResults.to_csv(rootFileDirectory + "Data/Results/Scale_Results_Without_Like_2.csv")
    
    
    SafeBoundParams = {benchmark : {'relativeErrorPerSegment' : .01,
                      'numHistogramBuckets' : 64,
                      'numEqualityOutliers' : 1028,
                      'numCDFGroups' :  32,
                      'trackNulls' : True,
                      'trackTriGrams' : True,
                      'numCores' : 8,
                      'groupingMethod' : "CompleteClustering",
                      'modelCDF' : True,
                      'verbose' : True} for benchmark in benchmarks
                      }
    
    buildTime = []
    statSize = []
    benchmarkDone = []
    hadTriGrams = []
    successful = []
    for benchmark in benchmarks:
        try:
            time, size = build_stats_object(method='SafeBound',
                               benchmark=benchmark,
                               parameters = {x:y for x,y in SafeBoundParams[benchmark].items()},
                               outputFile = SafeBoundFileName + benchmark + ".pkl"
                        )
            buildTime.append(time)
            statSize.append(size)
            benchmarkDone.append(benchmark)
            hadTriGrams.append(True)
            successful.append(True)
        except Exception as e:
            print(e)
            buildTime.append(0)
            statSize.append(0)
            benchmarkDone.append(benchmark)
            hadTriGrams.append(True)
            successful.append(False)
    safeBoundResults = pd.DataFrame()
    safeBoundResults['BuildTime'] = buildTime
    safeBoundResults['Size'] = statSize
    safeBoundResults['Benchmark'] = benchmarkDone
    safeBoundResults['HadTriGrams'] = hadTriGrams
    safeBoundResults['Successful'] = successful
    safeBoundResults.to_csv(rootFileDirectory + "Data/Results/Scale_Results_With_Like.csv")
    
    
    
    
    
    