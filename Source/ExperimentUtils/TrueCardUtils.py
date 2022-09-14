import pandas as pd
import pickle
from datetime import datetime, timedelta
import multiprocessing
import sys
import os
import pickle
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +'/'
sys.path.append(rootFileDirectory + "Source/")
from DBConnectionUtils import *
from SafeBoundUtils import *
from SQLParser import *
    
def get_cardinality(queryJG, queryHints, dbConn, result):
    estimate, actual = dbConn.getSizeEstimateAndActual(queryJG, queryHints)
    result.append(actual)
    
    
def gather_true_cardinalities(benchmark = 'JOBLight',
                              outputFile = '../../Data/Results/SafeBound_Inference_JOBLight_1.csv'):
    queryJGs = None        
    dbConn = None
    safeBound = pickle.load(open(rootFileDirectory + 'StatObjects/SafeBound_5_' + benchmark + ".pkl", "rb"))
    if benchmark == 'JOBLight':
        queryJGs =  SQLFileToJoinQueryGraphs(rootFileDirectory + 'Workloads/JOBLightQueries.sql')
        dbConn = getDBConn('imdblight')
    elif benchmark == 'JOBLightRanges':
        queryJGs =  SQLFileToJoinQueryGraphs(rootFileDirectory + 'Workloads/JOBLightRangesQueries.sql')
        dbConn = getDBConn('imdblightranges')
    elif benchmark == 'JOBM':
        queryJGs =  SQLFileToJoinQueryGraphs(rootFileDirectory + 'Workloads/JOBMQueries.sql')
        dbConn = getDBConn('imdbm')
    elif benchmark == 'Stats':
        queryJGs =  SQLFileToJoinQueryGraphs(rootFileDirectory + 'Workloads/StatsQueries.sql')
        dbConn = getDBConn('statsfk')
    
    allQueryHints = [[] for _ in range(len(queryJGs))]
    numPreviousHints = 0
    if os.path.exists(outputFile):
        allQueryHints = pickle.load(open(outputFile, "rb"))
        numPreviousHints = sum([len(x) > 0 for x in allQueryHints])
        if len(allQueryHints) < len(queryJGs):
            allQueryHints = allQueryHints[0:numPreviousHints] + [[] for _ in range(len(queryJGs) - numPreviousHints)]
    for i, queryJG in enumerate(queryJGs):
        print(i)
        if numPreviousHints > i:
            continue
        queryHints = []
        for subQuery in queryJG.getIntermediateQueries():
            print(i)
            subQueryHints = []
            for subSubQuery in subQuery.getIntermediateQueries():
                estimate = safeBound.functionalFrequencyBound(subSubQuery)
                tables = list(subSubQuery.vertexDict.keys())
                hint = JoinHint(tables, estimate)
                subQueryHints.append(hint)
            dbConn.printQueryPlan(subQuery, subQueryHints)
            manager = multiprocessing.Manager()
            result = manager.list()
            p = multiprocessing.Process(target=get_cardinality, args = [subQuery, subQueryHints, dbConn, result])
            p.start()
            p.join(5*60)
            print(safeBound.functionalFrequencyBound(subQuery))
            print(result)
            actual = 0
            if p.is_alive():
                actual = safeBound.functionalFrequencyBound(subQuery)
                p.kill()
                dbConn.reset()
            else:
                actual = result[0]
            tables = list(subQuery.vertexDict.keys())
            hint = JoinHint(tables, actual)
            queryHints.append(hint)
        allQueryHints[i] = queryHints
        pickle.dump(allQueryHints, open(outputFile, "wb"))
        if i % 10 == 1:
            print("True Card Gathering " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
    pickle.dump(allQueryHints, open(outputFile, "wb"))
