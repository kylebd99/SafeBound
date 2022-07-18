# cython: infer_types=True
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import pandas as pd
import itertools
import multiprocessing as mp
import concurrent.futures
import sys
rootFileDirectory = "/home/ec2-user/FrequencyBounds/"
sys.path.append(rootFileDirectory + 'Source')
from JoinGraphUtils import *
from HistogramUtils import *
from PiecewiseConstantFunctionUtils cimport *

class QueryTableStats:
    def __init__(self, alias, joinCols, colFuncs):
        self.alias = alias
        self.joinCols = joinCols
        self.colFuncs = colFuncs

class TableStats:
    
    def __init__(self, table, tableName, joinCols, relativeErrorPerSegment, filterCols = [],
                 numBuckets = 25, numEqualityOutliers = 500, numOutliers =5, trackNulls=True,
                 trackBiGrams=True, MPExecutor=None, groupingMethod="CompleteClustering", modelCDF=True, verbose=False):
        self.tableName = tableName
        self.relativeErrorPerSegment = relativeErrorPerSegment
        self.numOutliers = numOutliers
        self.numBuckets = numBuckets
        self.numEqualityOutliers = numEqualityOutliers
        self.filterCols = filterCols
        self.joinCols = joinCols
        self.trackNulls = trackNulls
        self.trackBiGrams = trackBiGrams
        self.histBins = dict()
        
        table = table[set(self.filterCols + self.joinCols)]
        
        self.functionHistogram = MultiColumnFunctionHistogram(table,
                                                              self.filterCols,
                                                              self.joinCols,
                                                              self.numBuckets,
                                                              self.numEqualityOutliers,
                                                              self.relativeErrorPerSegment, 
                                                              self.numOutliers,
                                                              self.trackNulls,
                                                              self.trackBiGrams,
                                                              MPExecutor,
                                                              groupingMethod,
                                                              modelCDF,
                                                              verbose)  

    def calculateError(self, func, data, offset = 0):
        sortedData = sorted(data, reverse = True)
        error = 0
        for i in range(len(sortedData)):
            error += abs(sortedData[i]-func.calculateValueAtPoint(offset + i))
        return error
    
    def getQueryTableStats(self, outputJoinCols, alias, preds, verbose):
        if verbose >= 2:
            print("Predicates for: " + self.tableName)
            for pred in preds:
                print(pred.toString())
            
        relevantColFuncs = dict()
        joinCols = outputJoinCols
        relevantColFuncs = self.functionHistogram.getMinFunctionsSatisfyingPredicates(preds, joinCols)
        for joinCol in joinCols:
            if verbose:
                print("Function for " + joinCol)
                relevantColFuncs[joinCol].printDiagnostics()

        minRows = 2**63
        for joinCol in joinCols:
            minRows = min(minRows, relevantColFuncs[joinCol].getNumRows())
        for joinCol in joinCols:
            relevantColFuncs[joinCol] = relevantColFuncs[joinCol].rightTruncateByRows(minRows)
        return QueryTableStats(alias, joinCols, relevantColFuncs)
            
    def printDiagnostics(self):
        print("Function Histgrams:")
        self.functionHistogram.printHists()
    
    def memory(self):
        return self.functionHistogram.memory()


class SafeBound:
    def __init__(self, tableDFs, tableNames, tableJoinCols, relativeErrorPerSegment, originalFilterCols = [], 
                     numBuckets = 25, numEqualityOutliers=500, FKtoKDict = dict(),
                     numOutliers = 5, trackNulls=True, trackBiGrams=True, numCores=12, groupingMethod = "CompleteClustering",
                     modelCDF=True, verbose=False):
        self.tableStatsDict = dict()
        self.relativeErrorPerSegment = relativeErrorPerSegment
        self.numOutliers = numOutliers
        self.numEqualityOutliers = numEqualityOutliers
        self.numBuckets = numBuckets
        self.numTables = len(tableDFs)
        self.trackNulls = trackNulls
        self.trackBiGrams = trackBiGrams
        self.tableNames = tableNames
        self.modelCDF = modelCDF
        
        if originalFilterCols == []:
            filterCols = [[] for _ in tableNames]
        else:
            filterCols = [x.copy() for x in originalFilterCols]
        
        tableNames = [x.upper() for x in tableNames]
        tableJoinCols = [x.copy() for x in tableJoinCols]
        self.universalFilterCols = [x.copy()  for x in filterCols]
        tableDFs = [tableDFs[i][list(set(filterCols[i] + tableJoinCols[i]))].copy() for i in range(len(tableDFs))]
        universalTableDFs = [x.copy() for x in tableDFs]
        
        # First, we rename all of the columns by prepending the table name to avoid confusion after the merging.
        # Additionally, we uppercase all names.
        for i in range(len(tableNames)):
            tableDFs[i].columns = [tableNames[i] + "." + x.upper() for x in tableDFs[i].columns]
            universalTableDFs[i].columns = [tableNames[i] + "." + x.upper() for x in universalTableDFs[i].columns]
            self.universalFilterCols[i] = [tableNames[i] +"."+ x.upper() for x in self.universalFilterCols[i]]
            tableJoinCols[i] = [tableNames[i] + "." + x.upper() for x in tableJoinCols[i]]
        
        FKtoKDictCopy = dict()
        self.FKtoKDict = dict()
        for leftTable, edges in FKtoKDict.items():
            newEdges = []
            for edge in edges:
                newEdges.append([leftTable.upper() + "."+ edge[0].upper(), edge[2].upper() + "." + edge[1].upper(), edge[2].upper()])
            self.FKtoKDict[leftTable.upper()] = newEdges.copy()
            FKtoKDictCopy[leftTable.upper()] = newEdges.copy()
        
        for _ in range(1):
            for i in range(len(tableNames)):
                tableName = tableNames[i]
                if tableName in FKtoKDictCopy:
                    edges = FKtoKDictCopy[tableName].copy()
                    newEdges = []
                    for edge in edges:
                        leftJoinCol = edge[0]
                        rightJoinCol = edge[1]
                        rightTableName = edge[2]
                        
                        for j in range(len(tableNames)):
                            if tableNames[j] == rightTableName:
                                filterColsToBeAdded = list(set(self.universalFilterCols[j])-set(self.universalFilterCols[i]))
                                joinColsToBeAdded = list(set(tableJoinCols[j])-set(tableJoinCols[i]))
                                colsToBeAdded = filterColsToBeAdded + joinColsToBeAdded
                                universalTableDFs[i] = universalTableDFs[i].merge(universalTableDFs[j][list(set([rightJoinCol]+colsToBeAdded))],
                                                                                  left_on=leftJoinCol, 
                                                                                  right_on=rightJoinCol,
                                                                                 how="left")
                                self.universalFilterCols[i].extend(filterColsToBeAdded)
                                self.universalFilterCols[i] = list(set(self.universalFilterCols[i]))
                                universalTableDFs[i] = universalTableDFs[i]
                                if rightTableName in FKtoKDictCopy:
                                    newEdges.extend(FKtoKDictCopy[rightTableName].copy())
                    FKtoKDictCopy[tableName] = newEdges.copy()
                    self.universalFilterCols[i] = list(set(self.universalFilterCols[i]))
        
        self.universalFilterColsDict = dict()
        for i in range(len(tableNames)):
            self.universalFilterColsDict[tableNames[i]] = self.universalFilterCols[i]
        
        tableStatsConstructorArgs = []
        if numCores > 1:
            # We can use a with statement to ensure threads are cleaned up promptly
            ctx = mp.get_context('spawn')
            MPExecutor = ctx.Pool(processes=numCores, maxtasksperchild=1)
            ThreadExecutor = concurrent.futures.ThreadPoolExecutor()
            futureToTableName = dict()
            for i in range(len(tableNames)):
                if verbose:
                    print("Building Table: " + tableNames[i])
                futureToTableName[ThreadExecutor.submit(TableStats,
                                                                  universalTableDFs[i],
                                                                  tableNames[i],
                                                                  tableJoinCols[i].copy(), 
                                                                  relativeErrorPerSegment,
                                                                  self.universalFilterCols[i].copy(),
                                                                  numBuckets,
                                                                  numEqualityOutliers,
                                                                  numOutliers,
                                                                  trackNulls,
                                                                  trackBiGrams,
                                                                  MPExecutor,
                                                                  groupingMethod,
                                                                  modelCDF,
                                                                  verbose)] = tableNames[i]
            for future in concurrent.futures.as_completed(futureToTableName):
                tableName = futureToTableName[future]
                try:
                    self.tableStatsDict[tableName] = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (tableName, exc))
        else:
            for i in range(len(tableNames)):
                if verbose:
                    print("Building Table: " + tableNames[i])
                self.tableStatsDict[tableNames[i]] = TableStats(universalTableDFs[i],
                                                                tableNames[i],
                                                                tableJoinCols[i].copy(), 
                                                                relativeErrorPerSegment,
                                                                self.universalFilterCols[i].copy(),
                                                                numBuckets,
                                                                numEqualityOutliers,
                                                                numOutliers,
                                                                trackNulls,
                                                                trackBiGrams,
                                                                None,
                                                                groupingMethod,
                                                                modelCDF,
                                                                verbose)
            
    def calculateNonJoiningColumnFrequency(self, PiecewiseConstantFunction RX, 
                                                 PiecewiseConstantFunction SX, 
                                                 PiecewiseConstantFunction SY):
        SY_to_X_RightX = []
        SY_to_X_RightY = []
        SX_to_Y_Slope = []
        cdef size_t idx0 = 0
        cdef size_t idx1 = 0
        finished = False
        while not finished:
            newRow = min(SX.cumulativeRows[idx0], SY.cumulativeRows[idx1])
            SY_to_X_RightX.append(SX.calculateInverse(newRow))
            SY_to_X_RightY.append(SY.calculateInverse(newRow))
            SX_to_Y_Slope.append(SX.constants[idx0]/SY.constants[idx1])
            if newRow >= SX.cumulativeRows[idx0]:
                idx0 += 1
            if newRow >= SY.cumulativeRows[idx1]:
                idx1 += 1
            if idx0 >= len(SX.cumulativeRows) or idx1 >= len(SY.cumulativeRows):
                finished = True
        
        if len(SY_to_X_RightX) == 0:
            return getEmptyFunction()

        finalConstants = []
        finalRightIntervalEdges = []
        idx0 = 0
        idx1 = 0
        finished = False
        while not finished:
            newXVal = min(RX.rightIntervalEdges[idx0], SY_to_X_RightX[idx1])
            leftY = 0
            leftX = 0
            if idx1 > 0:
                leftY = SY_to_X_RightY[idx1-1]
                leftX = SY_to_X_RightX[idx1-1]
            finalRightIntervalEdges.append(leftY + (newXVal - leftX) * SX_to_Y_Slope[idx1])
            finalConstants.append(RX.constants[idx0])
            if newXVal >= RX.rightIntervalEdges[idx0]:
                idx0 += 1
            if newXVal >= SY_to_X_RightX[idx1]:
                idx1 += 1
            if idx0 >= len(RX.rightIntervalEdges) or idx1 >= len(SY_to_X_RightX):
                finished = True
        
        if len(finalConstants) == 0:
            return RSY_TopFunc
        
        finalCumulativeRows = []
        curRow = 0
        curLeft = 0
        for i in range(len(finalRightIntervalEdges)):
            curRight = finalRightIntervalEdges[i]
            curRow += finalConstants[i]*(curRight - curLeft)
            finalCumulativeRows.append(curRow)
            curLeft = curRight

        finalFrequencyDist =  getEmptyFunction()
        finalFrequencyDist.setConstants(np.array(finalConstants, dtype='double'))
        finalFrequencyDist.setRightIntervalEdges(np.array(finalRightIntervalEdges, dtype="double"))
        finalFrequencyDist.setCumulativeRows(np.array(finalCumulativeRows, dtype="double"))
        return finalFrequencyDist
    
    def getRelevantPreds(self, joinQueryGraph, vertex, inPredsAliases, inPredsDerivedEqualsPreds):
        relevantFilterColsDict = dict()
        reachableAliases = set([vertex.alias])
        if vertex.tableName in self.FKtoKDict:
            FKEdges = self.FKtoKDict[vertex.tableName]
            for joinCol in vertex.outputJoinCols:
                FKEdgesForJoinCol = [x for x in FKEdges if x[0] == joinCol]
                equalityClassMembers = joinQueryGraph.equalityClasses[joinQueryGraph.equalityDict[vertex.alias+"."+joinCol]]
                for edge in FKEdgesForJoinCol:
                    for member in equalityClassMembers:
                        parts = member.split('.')
                        if edge[1] == (parts[1] + '.' + parts[2]) and edge[2] == joinQueryGraph.tableDict[parts[0]]:
                            reachableAliases.add(parts[0])
        
        relevantPreds = []
        for alias in reachableAliases:
            relevantPreds.extend(joinQueryGraph.vertexDict[alias].predicates)
        
        for i in range(len(inPredsAliases)):
            if inPredsAliases[i] in reachableAliases:
                relevantPreds.append(inPredsDerivedEqualsPreds[i])
        
        return relevantPreds
    
    def functionalFrequencyBound(self, joinQueryGraph, dbConn = None, verbose = 0):
        joinQueryGraph = joinQueryGraph.copy()
        joinQueryGraph.arbitrarilyBreakCycles()
        joinQueryGraph.addColumnSuffixes()
        
        aliasesList = []
        inPredsList = []
        for curVertex in joinQueryGraph.vertexDict.values():
            for pred in curVertex.predicates:
                if pred.predType == 'IN':
                    aliasesList.append(curVertex.alias)
                    inPredsList.append(pred)
            curVertex.predicates = [x for x in curVertex.predicates if x.predType != 'IN']
        equalsPredsLists = [[Predicate(x.colName, "=", y) for y in x.compValue] for x in inPredsList]
        
        aliases = joinQueryGraph.vertexDict.keys()
        rootAlias = None
        maxGeneration = len(aliases)
        for alias in aliases:
            maxGenerationGivenRoot = max(joinQueryGraph.getTopologicalGenerations(alias).values())
            if maxGenerationGivenRoot < maxGeneration:
                maxGeneration = maxGenerationGivenRoot
                rootAlias = alias
        
        inPredBoundSum = 0
        for inPredDerivedEqualsPreds in itertools.product(*equalsPredsLists):
            aliasJoinColFuncs = dict()
            for curVertex in joinQueryGraph.vertexDict.values():
                if dbConn != None:
                    try:
                        x = self.scanResultsCache.keys()
                    except:
                        self.scanResultsCache = dict()
                    if len(curVertex.predicates) == 0:
                        colFuncs = dict()
                        for joinCol in curVertex.outputJoinCols:
                            tableHistogram = self.tableStatsDict[curVertex.tableName].functionHistogram
                            colFuncs[joinCol] = tableHistogram.fullFunctions[joinCol]
                        aliasJoinColFuncs[curVertex.alias] = colFuncs
                    else:
                        tableKey = (curVertex.tableName, frozenset([pred.toString() for pred in curVertex.predicates]))
                        if tableKey in self.scanResultsCache:
                            aliasJoinColFuncs[curVertex.alias] = self.scanResultsCache[tableKey]
                        else:
                            colFuncs = dict()
                            for joinCol in self.tableStatsDict[curVertex.tableName].joinCols:
                                degreeSequence = dbConn.getDegreeSequence(curVertex.tableName, 
                                                                          joinCol, curVertex.predicates)
                                colFuncs[joinCol] =  PiecewiseConstantFunction(degreeSequence,
                                                                               self.relativeErrorPerSegment/4.0)
                            aliasJoinColFuncs[curVertex.alias] = colFuncs
                            self.scanResultsCache[tableKey] = colFuncs                            
                else:
                    relevantPreds = self.getRelevantPreds(joinQueryGraph,
                                                          curVertex, 
                                                          aliasesList,
                                                          inPredDerivedEqualsPreds)
                    curQueryStats = self.tableStatsDict[curVertex.tableName].getQueryTableStats(
                                                                                        list(set(curVertex.outputJoinCols)),
                                                                                        curVertex.alias,
                                                                                        relevantPreds,
                                                                                        verbose)
                    aliasJoinColFuncs[curVertex.alias] = curQueryStats.colFuncs
            
            aliasToGeneration = joinQueryGraph.getTopologicalGenerations(rootAlias)
            for curGeneration in reversed(range(0, maxGeneration)):
                childGeneration = curGeneration + 1
                currentGenAliases = [x for x in aliases if aliasToGeneration[x] == curGeneration]
                for alias in currentGenAliases:
                    childIndices = [i for i, x in enumerate(joinQueryGraph.vertexDict[alias].edgeAliases) if aliasToGeneration[x] == childGeneration]
                    childAliases = [x for x in joinQueryGraph.vertexDict[alias].edgeAliases if aliasToGeneration[x] == childGeneration]
                    childJoinCols = [joinQueryGraph.vertexDict[alias].edgeJoinCols[i] for i in childIndices]
                    joinColsToChildren = [joinQueryGraph.vertexDict[alias].outputJoinCols[i] for i in childIndices]
                    joinColsToChildJoinCol = {x:[] for x in set(joinColsToChildren)}
                    for i in range(len(joinColsToChildren)):
                        joinColsToChildJoinCol[joinColsToChildren[i]].append(aliasJoinColFuncs[childAliases[i]][childJoinCols[i]])
                    joinColToParent = None 
                    joinColToParentFunc = None
                    functionsToMultiply = []
                    if curGeneration > 0:
                        joinColToParent = [x for i, x in enumerate(joinQueryGraph.vertexDict[alias].outputJoinCols)
                                           if i not in childIndices][0]
                        joinColToParentFunc =  aliasJoinColFuncs[alias][joinColToParent]
                    else:
                        joinColToParentFunc = list(aliasJoinColFuncs[alias].values())[0]
                    functionsToMultiply = [joinColToParentFunc]
                    for joinColToChild, childJoinFuncs in joinColsToChildJoinCol.items():
                        joinColToChildFunc = aliasJoinColFuncs[alias][joinColToChild]
                        multipliedChildJoinFuncs = pointwiseFunctionMult(np.array(childJoinFuncs))
                        functionsToMultiply.append(self.calculateNonJoiningColumnFrequency(multipliedChildJoinFuncs,
                                                                                           joinColToChildFunc,
                                                                                           joinColToParentFunc))
                    resultFunc = pointwiseFunctionMult(np.array(functionsToMultiply))
                    if curGeneration > 0:
                        aliasJoinColFuncs[alias][joinColToParent] = resultFunc
                    else:
                        bound = resultFunc.getNumRows()
                        inPredBoundSum += bound
        return inPredBoundSum
    
    def printDiagnostics(self):
        print("PRINTING DIAGNOSTICS")
        
        for name, tableStat in self.tableStatsDict.items():
            print("Stats for: " + name)
            tableStat.printDiagnostics()
            
    def memory(self):
        footprint = 0
        for table, tableStats in self.tableStatsDict.items():
            print("Table: " + table)
            footprint += tableStats.memory()
        return footprint