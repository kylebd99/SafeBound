# cython: infer_types=True
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import pandas as pd
import itertools
import multiprocessing as mp
import concurrent.futures
import sys,os
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
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
    def __init__(self, table, tableName, joinCols, filterCols, dimTables, dimJoins, dimFilterCols, relativeErrorPerSegment = .05,
                 numBins = 25, numEqualityOutliers = 500, numOutliers = 5, trackNulls = True,
                 trackTriGrams = True, groupingMethod = "CompleteClustering", modelCDF = True, verbose = False):
        self.tableName = tableName
        self.numRows = len(table)
        self.numBins = numBins
        self.numEqualityOutliers = numEqualityOutliers
        self.relativeErrorPerSegment = relativeErrorPerSegment
        self.numOutliers = numOutliers
        self.filterCols = filterCols.copy()
        self.joinCols = joinCols.copy()
        self.fullFunctions = dict()           
        self.filterColStats = dict()
        self.trackNulls = trackNulls
        self.trackTriGrams = trackTriGrams
        self.groupingMethod = groupingMethod
        self.modelCDF = modelCDF
        self.verbose = verbose
        
        
        if verbose > 0:
            print("Building Full Table Approximations")
        
        for joinCol in self.joinCols:
            valCounts = np.array(table[joinCol].value_counts(ascending=False).to_list(), dtype='int')
            function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment/4.0, modelCDF=modelCDF) # The number of segments is increased for the full approximation. 
            function.compressFunc()
            self.fullFunctions[joinCol] = function
        
        self.table = table
        self.filterColAndDimTableJoin = []
        for filterCol in filterCols:
            self.filterColAndDimTableJoin.append([filterCol, None])
        
        for i in range(len(dimTables)):
            for filterCol in dimFilterCols[i]:
                self.filterColAndDimTableJoin.append([filterCol, (dimJoins[i][0], dimJoins[i][1], dimTables[i])])
            
        
    def startFilterColumnStatsBuildProcess(self, MPExecutor):
        if len(self.filterColAndDimTableJoin) == 0:
            self.table = None
            return None
        
        curFilterCol, curDimTableJoin = self.filterColAndDimTableJoin.pop(0)
        curTable = self.table
        if curDimTableJoin:
            curTable = self.table.merge(curDimTableJoin[2][[curFilterCol, curDimTableJoin[1]]] , left_on=curDimTableJoin[0], right_on=curDimTableJoin[1])
        
        if self.verbose > 0:
            print("Building Stats: " + curFilterCol)
        futureStats = MPExecutor.submit(FilterColumnStats, 
                                              curTable[list(set([curFilterCol]+self.joinCols))], 
                                              curFilterCol,
                                              self.joinCols.copy(),
                                              self.numBins,
                                              self.numEqualityOutliers, 
                                              self.relativeErrorPerSegment,
                                              self.numOutliers,
                                              self.trackNulls,
                                              self.trackTriGrams,
                                              self.groupingMethod,
                                              self.modelCDF,
                                              self.verbose)
        return curFilterCol, futureStats
        
    def addFilterColStats(self, filterCol, stats):
        self.filterColStats[filterCol] = stats
    
    def getMinFunctionsSatisfyingPredicates(self, listOfPreds, joinCols):
        if self.numRows == 0:
            return {joinCol: getEmptyFunction() for joinCol in joinCols}
        
        numValidPreds = 0
        predsByColumn = dict()
        for pred in listOfPreds:
            predsByColumn[pred.colName] = []
        for pred in listOfPreds:
            if pred.predType in [">",">=","<=","<"] and self.filterColStats[pred.colName].numBins == 0:
                continue
            elif pred.predType == "LIKE" and self.filterColStats[pred.colName].trackTriGrams == False:
                continue
            elif pred.predType in ["IS NULL", "IS NOT NULL"]  and self.filterColStats[pred.colName].trackNulls == False:
                continue
            predsByColumn[pred.colName].append(pred)
            numValidPreds += 1
        
        if numValidPreds == 0:
            minFunctionsDict = dict()
            for joinCol in joinCols:
                minFunctionsDict[joinCol] = self.fullFunctions[joinCol]
            return minFunctionsDict
        
        validFunctions = [[] for _ in joinCols]
        for filterCol, preds in predsByColumn.items():
            joinColFunctions = self.filterColStats[filterCol].getMinFunctionsSatisfyingPredicates(preds, joinCols)
            for i in range(len(joinCols)):
                joinCol = joinCols[i]
                function = joinColFunctions[joinCol]
                validFunctions[i].append(function)
    
        minFunctionsDict = dict()
        for i in range(len(joinCols)):
            joinCol = joinCols[i]
            minFunctionsDict[joinCol] = pointwiseFunctionMin(np.array(validFunctions[i]))
        return minFunctionsDict
    
    def getQueryTableStats(self, outputJoinCols, alias, preds, verbose):
        if verbose >= 2:
            print("Predicates for: " + self.tableName)
            for pred in preds:
                print(pred.toString())
            
        relevantColFuncs = dict()
        joinCols = outputJoinCols
        relevantColFuncs = self.getMinFunctionsSatisfyingPredicates(preds, joinCols)
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
        for col in self.filterCols:
            print("Stats For Column: " + col)
            self.filterColStats[col].printDiagnostics()
        
    def memory(self):
        footprint = 0
        for col, stats in self.filterColStats.items():
            footprint += stats.memory()
        for joinCol in self.joinCols:
            footprint += self.fullFunctions[joinCol].memory()
        return footprint


class SafeBound:
    def __init__(self, tableDFs, tableNames, tableJoinCols, relativeErrorPerSegment, originalFilterCols = [], 
                     numBuckets = 25, numEqualityOutliers=500, FKtoKDict = dict(),
                     numOutliers = 5, trackNulls=True, trackTriGrams=True, numCores=12, groupingMethod = "CompleteClustering",
                     modelCDF=True, verbose=False):
        self.tableStatsDict = dict()
        self.relativeErrorPerSegment = relativeErrorPerSegment
        self.numOutliers = numOutliers
        self.numEqualityOutliers = numEqualityOutliers
        self.numBuckets = numBuckets
        self.numTables = len(tableDFs)
        self.trackNulls = trackNulls
        self.trackTriGrams = trackTriGrams
        self.tableNames = tableNames
        self.modelCDF = modelCDF
        
        if originalFilterCols == []:
            filterCols = [[] for _ in tableNames]
        else:
            filterCols = [x.copy() for x in originalFilterCols]
        
        self.tableNames = [x.upper() for x in self.tableNames]
        tableJoinCols = [x.copy() for x in tableJoinCols]
        
        # First, we rename all of the columns by prepending the table name to avoid confusion after the merging.
        # Additionally, we uppercase all names.
        for i in range(len(tableNames)):
            tableDFs[i].columns = [self.tableNames[i] + "." + x.upper() for x in tableDFs[i].columns]
            filterCols[i] = [self.tableNames[i] +"."+ x.upper() for x in filterCols[i]]
            tableJoinCols[i] = [self.tableNames[i] + "." + x.upper() for x in tableJoinCols[i]]
        self.universalFilterCols = [x.copy()  for x in filterCols]
        
        self.FKtoKDict = dict()
        for leftTable, edges in FKtoKDict.items():
            newEdges = []
            for edge in edges:
                newEdges.append([leftTable.upper() + "."+ edge[0].upper(), edge[2].upper() + "." + edge[1].upper(), edge[2].upper()])
            self.FKtoKDict[leftTable.upper()] = newEdges.copy()
        
        ctx = mp.get_context('spawn')
        MPExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=numCores, mp_context=ctx)
        for i in range(len(self.tableNames)):
            tableName = self.tableNames[i]
            dimTables = []
            dimJoins = []
            dimFilterCols = []
            if tableName in self.FKtoKDict:
                dimJoins = self.FKtoKDict[tableName]
                for join in dimJoins:
                    for j in range(len(self.tableNames)):
                        if join[2] == self.tableNames[j]:
                            dimTables.append(tableDFs[j])
                            dimFilterCols.append(filterCols[j])
                dimJoins = [(x[0], x[1]) for x in dimJoins]
            self.tableStatsDict[self.tableNames[i]] = TableStats(tableDFs[i],
                                                              self.tableNames[i],
                                                              tableJoinCols[i].copy(),
                                                              self.universalFilterCols[i], 
                                                              dimTables,
                                                              dimJoins,
                                                              dimFilterCols,
                                                              relativeErrorPerSegment,
                                                              numBuckets,
                                                              numEqualityOutliers,
                                                              numOutliers,
                                                              trackNulls,
                                                              trackTriGrams,
                                                              groupingMethod,
                                                              modelCDF,
                                                              verbose)
        finishedBuildingStats = False
        self.activeTablesAndFilterColumnsAndFutures = []
        while not finishedBuildingStats or len(self.activeTablesAndFilterColumnsAndFutures) > 0:
            while not finishedBuildingStats and len(self.activeTablesAndFilterColumnsAndFutures) < numCores + 1:
                tableAndFilterColumnAndFuture = self.startOneStatsBuildProcess(MPExecutor)
                if tableAndFilterColumnAndFuture:
                    self.activeTablesAndFilterColumnsAndFutures.append(tableAndFilterColumnAndFuture)
                else:
                    finishedBuildingStats = True
                    break
            self.finishOneStatsBuildProcess()
    
    def finishOneStatsBuildProcess(self):
        finished = False
        while not finished:
            for i in range(len(self.activeTablesAndFilterColumnsAndFutures)):
                if self.activeTablesAndFilterColumnsAndFutures[i][2].done():
                    tableName, filterCol, statsFuture = self.activeTablesAndFilterColumnsAndFutures.pop(i)
                    self.tableStatsDict[tableName].addFilterColStats(filterCol, statsFuture.result())
                    finished = True
                    break
            time.sleep(.1)
    
    def startOneStatsBuildProcess(self, MPExecutor):
        for table in self.tableNames:
            filterColAndStatsFuture = self.tableStatsDict[table].startFilterColumnStatsBuildProcess(MPExecutor)
            if filterColAndStatsFuture:
                return (table, filterColAndStatsFuture[0], filterColAndStatsFuture[1])
        return None
    
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