# cython: infer_types=True, boundscheck=False, language='c++'
cimport openmp
from cython.parallel import prange
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.set cimport set as c_set
import sys
import pandas as pd
import numpy as np
import gc
import psutil
from math import ceil, log2, sqrt
from bisect import bisect_left, bisect_right
from itertools import combinations
from collections import Counter
import concurrent.futures
import multiprocessing as mp
import fastcluster as fc
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from pybloomfilter import BloomFilter
import os
import sys
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory + 'Source')
from JoinGraphUtils import *
from PiecewiseConstantFunctionUtils cimport *

def memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/1000000000.

def getQuantiles(values, numBins, numVals, includeDuplicates = True, alreadySorted=False):
    quantiles = []
    sortedValues = values
    if not alreadySorted:
        sortedValues = list(sorted(values))
    prevVal = None
    for i in range(1, numBins+1):
        idx = int(len(values)*(float(i)/(numBins+1)))
        nextVal = sortedValues[min(idx, len(sortedValues)-1)]
        quantiles.append(nextVal)
    if includeDuplicates == False:
        if numVals <= numBins:
            return sorted(list(set(values)))
        quantiles = sorted(list(set(quantiles)))
        binsToAdd = numBins - len(quantiles)
        valuesSeries = pd.Series(values)
        remainingValuesSeries = valuesSeries[~valuesSeries.isin(quantiles)]
        topRemainingValues = remainingValuesSeries.value_counts().head(binsToAdd).reset_index()["index"]
        if len(topRemainingValues) > 0 and binsToAdd > 0:
            quantiles = sorted(quantiles + list(topRemainingValues))
    return quantiles

cdef class TriGramSet:
    cdef c_set[string] triGrams
    
    def __init__(self, triGrams):
        self.triGrams = triGrams
    
    def __contains__(self, string x):
        return self.triGrams.count(x) > 0
    
    def __sizeof__(self):
        return sum([x.__sizeof__() for x in self.triGrams])
    
    

# This object represents the statistics held for a particular filter-able column. Broadly, this includes a histogram for range queries,
# a most-common value list for equality queries, and a tri-gram list for LIKE queries. Within these statistics, there are degree sequences
# stored for the join columns of the table which can be retrieved at query time for bound generation.
class FilterColumnStats:

    def __init__(self, table, filterCol, joinCols, numBins, numEqualityOutliers, relativeErrorPerSegment, 
                 numOutliers, trackNulls, trackTriGrams, groupingMethod, modelCDF, verbose):
        self.numValues = len(table)
        self.filterCol = filterCol
        self.joinCols = joinCols
        self.relativeErrorPerSegment = relativeErrorPerSegment
        numBins = min(numBins, self.numValues)
        if numBins > 0:
            numBins = 2**ceil(log2(numBins))
            self.numLevels = int(log2(numBins))+1
        else:
            self.numLevels = 0
        self.numBins = numBins
        self.numEqualityOutliers = numEqualityOutliers
        self.numOutliers = numOutliers
        self.trackNulls = trackNulls
        self.trackTriGrams = trackTriGrams
        
        self.isStringColumn = table[filterCol].dtype == 'object'
        notNullTable = table.loc[~table[filterCol].isna()]
        nullTable = table.loc[table[filterCol].isna()]
        del table
        notNullTable.sort_values(filterCol, inplace=True)
        if self.trackNulls:
            if verbose > 0:
                print("Approximating null distributions: "+ filterCol)
            # Record the distributions of the join columns for Is Null and Is Not Null Predicates.
            self.nullFunctions = dict()
            for joinCol in joinCols:
                valCounts = np.array(nullTable[joinCol].value_counts(ascending=False).to_list(), dtype='int')
                if len(valCounts) == 0:
                    self.nullFunctions[joinCol] = getEmptyFunction()
                else:
                    function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                    function.compressFunc()
                    self.nullFunctions[joinCol] = function
            nullTable = None
            
            if verbose > 0:
                print("Approximating not null distributions: "+ filterCol)
            self.notNullFunctions = dict()
            for joinCol in joinCols:
                valCounts = np.array(notNullTable[joinCol].value_counts(ascending=False).to_list(), dtype='int')
                if len(valCounts) == 0:
                    self.notNullFunctions[joinCol] = getEmptyFunction()
                else:
                    function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                    function.compressFunc()
                    self.notNullFunctions[joinCol] = function
            del valCounts
        del nullTable
        gc.collect()
        
        self.notNullValues = len(notNullTable)
        if self.notNullValues == 0:
            return
        
        self.min = min(notNullTable[filterCol])
        if self.isStringColumn:
            startingVal = ""
        else:
            startingVal = min(notNullTable[filterCol])-1
        self.max = max(notNullTable[filterCol])
        
        
        if self.numBins > 0:
            if verbose > 0:
                print("Approximating range distributions: "+ filterCol+ " Memory Usage: " + str(memory()))
            # For the rest of the predicates, we screen out all null values and use notNullTable
            self.rightEdges = getQuantiles(notNullTable[filterCol].tolist(),
                                           numBins,
                                           notNullTable[filterCol].nunique(),
                                           includeDuplicates=False,
                                           alreadySorted=True)

            self.stepSizes = [2**k for k in range(self.numLevels)]
            self.intervals = []
            for stepSize in self.stepSizes:
                left = startingVal
                for i in range(stepSize-1, len(self.rightEdges), stepSize):
                    right = self.rightEdges[i]
                    self.intervals.append((left, right, stepSize))
                    left = right
                if left < self.max:
                    self.intervals.append((left, self.max, stepSize))
                numStartingPoints = 2
                if stepSize >= 4:
                    for startingPoint in [x*int(stepSize/numStartingPoints) for x in range(1,numStartingPoints)]:
                        if startingPoint >= len(self.rightEdges):
                            continue
                        left = self.rightEdges[startingPoint]
                        for i in range(startingPoint+stepSize, len(self.rightEdges), stepSize):
                            right = self.rightEdges[i]
                            self.intervals.append((left, right, stepSize))
                            left = right
                        if left < self.max:
                            self.intervals.append((left, self.max, stepSize))
            intervalDict = dict()
            for interval in self.intervals:
                if (interval[0], interval[1]) in intervalDict:
                    intervalDict[(interval[0],interval[1])] = min(interval[2], intervalDict[(interval[0],interval[1])])
                else:
                    intervalDict[(interval[0],interval[1])] = interval[2]
            self.intervals = []
            for interval, stepSize in intervalDict.items():
                self.intervals.append((interval[0], interval[1]))
            self.intervals.append((startingVal, self.max, self.numValues))
            self.stepSizes.append(self.numValues)


            # We calculate each interval's functional approximation for each join column. Then, we sort those approximations
            # via a rough scoring method to determine a grouping. Each grouping is then represented by the pointwise max of the
            # functions in that group.        

            if verbose > 1:
                print("Generating Range Functions: "+ filterCol + " Memory Usage: " + str(memory()))
            nonEdgeIntervals = []
            nonEdgeIntervalFunctionDicts = []
            edgeIntervals = []
            edgeIntervalDicts = [] 
            for interval in self.intervals:
                leftIdx = bisect_right(notNullTable[filterCol].array, interval[0])
                rightIdx = max(bisect_right(notNullTable[filterCol].array, interval[1]), leftIdx + 1)
                functionDict = dict()
                for joinCol in self.joinCols:
                    valCounts = np.array(notNullTable[joinCol].iloc[leftIdx:rightIdx].value_counts(ascending=False).to_list(), dtype='int')
                    function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                    function.compressFunc()
                    functionDict[joinCol] = function
                if interval[1] == self.rightEdges[-1]  or interval[0] == startingVal:
                    edgeIntervals.append(interval)
                    edgeIntervalDicts.append(functionDict)
                else:
                    nonEdgeIntervals.append(interval)
                    nonEdgeIntervalFunctionDicts.append(functionDict)
            groupIntervals, self.representativeRangeFunctionDicts = FilterColumnStats.coalesceFunctionDicts(nonEdgeIntervals, 
                                                                                                              nonEdgeIntervalFunctionDicts, 
                                                                                                              joinCols,
                                                                                                              self.numOutliers,
                                                                                                              groupingMethod,
                                                                                                              modelCDF)
            del valCounts
            
            
            for i in range(len(edgeIntervals)):
                groupIntervals.append([edgeIntervals[i]])
                self.representativeRangeFunctionDicts.append(edgeIntervalDicts[i])
            
            self.intervalToFunctionDictIdx = dict()
            for i in range(len(groupIntervals)):
                for interval in groupIntervals[i]:
                    self.intervalToFunctionDictIdx[interval] = i
            del self.intervals
        
        gc.collect()
        # In order to not calculate a function for every filter value X join col pair, we are instead going to do some approximation.
        # First, we take a rough selection of outlier distributions based on the row-count of the equality predicates. For high row-count
        # predicates, we calculate their function approximations then do a similar grouping as in the intervals above.
        # We represent each grouping via a bloom filter to save space and speed querying. The remainder are all grouped and maxed as
        # one function using batch methods in pandas.
        if verbose > 0:
            print("Approximating equality distributions: "+ filterCol+ " Memory Usage: " + str(memory()))
        self.equalityMaxFunctions = dict()
        
        numFilterValues = notNullTable[filterCol].nunique()

        filterOutlierValCounts = notNullTable[filterCol].value_counts().head(self.numEqualityOutliers).reset_index()
        outlierVals = filterOutlierValCounts["index"]
        outlierCounts = filterOutlierValCounts[filterCol]
        
        outlierFunctionDicts = []
        for curVal in outlierVals:
            leftIdx = bisect_left(notNullTable[filterCol].array, curVal)
            rightIdx = bisect_right(notNullTable[filterCol].array, curVal)
            functionDict = dict()
            for joinCol in joinCols:
                valCounts = np.array(notNullTable[joinCol].iloc[leftIdx:rightIdx].value_counts(ascending=False).to_list(), 
                                     dtype='int')
                function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                function.compressFunc()
                functionDict[joinCol] = function
            outlierFunctionDicts.append(functionDict)
        
        self.bloomFilters = []
        groupValues, self.representativeEqualityFunctionDicts = FilterColumnStats.coalesceFunctionDicts(outlierVals,
                                                                                                          outlierFunctionDicts, 
                                                                                                          joinCols,
                                                                                                          self.numOutliers,
                                                                                                          groupingMethod,
                                                                                                          modelCDF)
        del valCounts
        
        for i in range(len(groupValues)):
            vals = groupValues[i]
            bloomFilter = BloomFilter(len(vals), .00001)
            for val in vals:
                if isinstance(val, str):
                    bloomFilter.add(val)
                else:
                    bloomFilter.add(str(float(val)))
            self.bloomFilters.append(bloomFilter)
        
        # To handle the remaining equality values, Groupby into [filterCol, size], 
        # then rank these pairs by size within each filterCol value. Then,take the maximum over each rank. 
        # This gives you the pointwise maximum degree function. Lastly, truncate it to the maximum number of
        # rows per filter value.
        remainderTable = notNullTable[~notNullTable[filterCol].isin(outlierVals)]
        remainderMaxCount = remainderTable.groupby(filterCol).size().max()
        remainderMaxDegrees = None
        remainderFilterJoinAndSize = None
        for joinCol in joinCols:
            if len(remainderTable) == 0:
                self.equalityMaxFunctions[joinCol] = getEmptyFunction()
                continue
            
            if filterCol != joinCol:
                remainderFilterJoinAndSize = remainderTable.groupby([filterCol, joinCol]).size().reset_index(name="size")
                remainderFilterJoinAndSize['rank'] = remainderFilterJoinAndSize.groupby(filterCol)['size'].rank(method='first')
                remainderMaxDegrees = remainderFilterJoinAndSize.groupby('rank')['size'].max().sort_values(ascending=False)
            else:
                remainderMaxDegrees = remainderTable.groupby(filterCol).size().sort_values(ascending=False).head(1)
            maxFunction = PiecewiseConstantFunction(remainderMaxDegrees.to_numpy(), self.relativeErrorPerSegment, modelCDF=modelCDF)
            maxFunction = maxFunction.rightTruncateByRows(remainderMaxCount)
            maxFunction.compressFunc()
            self.equalityMaxFunctions[joinCol] = maxFunction
        del remainderTable
        del remainderFilterJoinAndSize
        del remainderMaxDegrees
        gc.collect()
        
        
        cdef vector[string] filterValues
        cdef int numValues = len(notNullTable)
        cdef int valIdx, mctgIdx, charIdx
        cdef map[string, vector[int]] mostCommonTriGramsToIndexes
        cdef map[string, int] mostCommonTriGramsToLockIdx
        cdef string valueTriGram
        cdef openmp.omp_lock_t[100]  mylocks
        for i in range(100):
            openmp.omp_init_lock(&mylocks[i]) # initialize
        # We handle string columns separately because they may have LIKE predicates applied to them.
        # Current Status: look into the str vs string situation
        if self.trackTriGrams and self.isStringColumn:
            
            if verbose > 0:
                print("Detecting Most Common TriGrams: "+ filterCol + " " + str(len(notNullTable)) + " Memory Usage: " + str(memory()))
            triGramCol = filterCol + "_TriGrams"
            triGramCounts = Counter([triGram for x in notNullTable[filterCol].sample(min(len(notNullTable)-1, 100000)) for triGram in self.getTriGramsFromString(x)])
            mostCommonTriGrams = [x[0] for x in triGramCounts.most_common(self.numEqualityOutliers)]
            mostCommonTriGramsToIndexes = {bytes(x[0], encoding='utf8') : np.array([], dtype='int32') for x in triGramCounts.most_common(self.numEqualityOutliers)}
            mostCommonTriGramsToLockIdx = {bytes(x[0], encoding='utf8') : i % 100 for i, x in enumerate(triGramCounts.most_common(self.numEqualityOutliers))}
            del triGramCounts
            filterValues = [bytes(x, encoding='utf8') for x in notNullTable[filterCol]]
            for valIdx in prange(numValues, nogil=True, schedule="static", chunksize=100000, num_threads=8):
                if filterValues[valIdx].size() < 3:
                    continue
                for charIdx in range(max(0, filterValues[valIdx].size()-2)):
                    valueTriGram = filterValues[valIdx].substr(charIdx, 3)
                    if mostCommonTriGramsToIndexes.count(valueTriGram) > 0:
                        openmp.omp_set_lock(&mylocks[mostCommonTriGramsToLockIdx[valueTriGram]]) 
                        mostCommonTriGramsToIndexes[valueTriGram].push_back(valIdx)
                        openmp.omp_unset_lock(&mylocks[mostCommonTriGramsToLockIdx[valueTriGram]]) 
            for i in range(100):
                openmp.omp_destroy_lock(&mylocks[i]) 
            mostCommonTriGramsToLockIdx.clear()
            filterValues.clear()
            
            if verbose > 0:
                print("Creating Function Approximations For Most Common TriGrams: "+ filterCol + " " + str(len(notNullTable))+ " Memory Usage: " + str(memory()))
            outlierFunctionDicts = []
            outlierTriGrams = []
            for i, triGram in enumerate(mostCommonTriGrams):
                subTable = notNullTable.iloc[mostCommonTriGramsToIndexes[bytes(triGram, encoding="utf8")]]
                if len(subTable) == 0:
                    continue
                functionDict = dict()
                for joinCol in joinCols:
                    valCounts = np.array(subTable[joinCol].value_counts(ascending=False).to_list(), dtype='int')
                    function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                    function.compressFunc()
                    functionDict[joinCol] = function
                outlierFunctionDicts.append(functionDict)
                outlierTriGrams.append(triGram)
            mostCommonTriGramsToIndexes.clear()
            
            
            if verbose > 0:
                print("Clustering Function Approximations For Most Common TriGrams: "+ filterCol + " " + str(len(notNullTable))+ " Memory Usage: " + str(memory()))
            groupValues, self.representativeTriGramFunctionDicts = FilterColumnStats.coalesceFunctionDicts(outlierTriGrams,
                                                                                                              outlierFunctionDicts, 
                                                                                                              joinCols,
                                                                                                              self.numOutliers,
                                                                                                              groupingMethod,
                                                                                                              modelCDF)
            self.outlierTriGramGroupSets = []
            for i, triGrams in enumerate(groupValues):
                self.outlierTriGramGroupSets.append(set(triGrams))
            
            
            if verbose > 0:
                print("Outlier Group Sets: "+ filterCol + " " + str(len(self.outlierTriGramGroupSets)))  
            
            
            # If no outlier TriGrams are present in the query string, we return the degree sequence of all rows
            # which contain none of the outlier TriGrams
            if verbose > 0:
                print("Handling Remainder Rows: "+ filterCol + " " + str(len(notNullTable))+ " Memory Usage: " + str(memory()))    
            mostCommonTriGramsSet = set(mostCommonTriGrams)
            notNullTable = notNullTable[notNullTable[filterCol].apply(lambda x: all([y not in mostCommonTriGramsSet for y in \
                                                                              self.getTriGramsFromString(x)]))]
            self.triGramRemainderFunctionDict = dict()
            for joinCol in joinCols:
                valCounts = np.array(notNullTable[joinCol].value_counts(ascending=False).to_list(), dtype='int')
                function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                function.compressFunc()
                self.triGramRemainderFunctionDict[joinCol] = function
        gc.collect()
        if verbose > 0:
            print("Finished Building Stats: " + filterCol)
    
    @staticmethod
    def coalesceFunctionDicts(values, functionDicts, joinCols, numResultFunctionDicts, groupingMethod, modelCDF):        
        if len(values) < max(5, numResultFunctionDicts):
            return [[x] for x in values], functionDicts
        
        if groupingMethod == "Naive":
            sortedValues, sortedFunctionDicts = zip(*sorted(zip(values, functionDicts), key=lambda x : list(x[1].values())[0].getNumRows()))
            resultValues = []
            resultFunctionDicts = []
            for i in range(numResultFunctionDicts):
                left = int(float(i)/numResultFunctionDicts * len(functionDicts))
                right =int(float(i+1)/numResultFunctionDicts * len(functionDicts))
                if left == right:
                    continue
                resultValues.append(sortedValues[left:right])
                resultFunctionDicts.append({joinCol: pointwiseFunctionMax(np.array([x[joinCol] for x in sortedFunctionDicts[left:right]])) for joinCol in joinCols})
            return resultValues, resultFunctionDicts
        
        # cdef definitions must occur outside of any control flow (e.g. for), so we prepare all statements here in order to use them in the parallel section
        # which follows
        maxLen = max([max([len(functionDicts[i][joinCol].rightIntervalEdges) for joinCol in joinCols]) for i in range(len(functionDicts))])
        cdef double[:, :, :] rightIntervalEdges = np.array([[[functionDicts[i][joinCol].rightIntervalEdges[j] if j < len(functionDicts[i][joinCol].rightIntervalEdges) else 0 \
                                                              for j in range(maxLen)] for joinCol in joinCols] for i in range(len(functionDicts))], dtype=np.double)
        cdef double[:, :, :] cumulativeRows = np.array([[[functionDicts[i][joinCol].rightIntervalEdges[j] if j < len(functionDicts[i][joinCol].rightIntervalEdges) else 0 \
                                                          for j in range(maxLen)] for joinCol in joinCols] for i in range(len(functionDicts))], dtype=np.double)
        cdef int[:, :] idxLen = np.array([[len(functionDicts[i][joinCol].rightIntervalEdges) for joinCol in joinCols] for i in range(len(functionDicts))], dtype=np.int32)
        cdef double[:] selfJoinSizes = np.array([sum([x[joinCol].getSelfJoinSize() for joinCol in joinCols]) for x in functionDicts], dtype=np.double)
        cdef double[:, :] distanceMatrix = np.zeros((len(functionDicts), len(functionDicts)), dtype=np.double)
        cdef int i1,j1,k1
        for i1 in prange(rightIntervalEdges.shape[0], num_threads= 8, nogil=True):
            for j1 in range(i1+1, rightIntervalEdges.shape[0]):
                distanceMatrix[i1][j1] = 0
#                for k1 in range(rightIntervalEdges.shape[1]):
#                    distanceMatrix[i1][j1] = distanceMatrix[i1][j1] + pairwiseMaxAndSelfJoin(rightIntervalEdges[i1][k1],
#                                                                                             rightIntervalEdges[j1][k1],
#                                                                                             cumulativeRows[i1][k1],
#                                                                                             cumulativeRows[j1][k1],
#                                                                                             idxLen[i1][k1],
#                                                                                             idxLen[j1][k1])
                distanceMatrix[i1][j1] = max(selfJoinSizes[i1],selfJoinSizes[j1])/min(selfJoinSizes[i1], selfJoinSizes[j1]) # Canberra Distance
                distanceMatrix[j1][i1] = distanceMatrix[i1][j1]
        
        
        squareDistanceMatrix = squareform(distanceMatrix)
        dendrogram = None
        if groupingMethod == "CompleteClustering":
            dendrogram =  fc.linkage(squareDistanceMatrix, method='complete')
        elif groupingMethod == "SingleClustering":
            dendrogram =  fc.linkage(squareDistanceMatrix, method='single')
        clusters = sch.fcluster(dendrogram, numResultFunctionDicts, criterion='maxclust') - 1 # Subtract 1 so that group labels match indices 
        resultFunctionDicts = [{joinCol:[] for joinCol in joinCols} for _ in range(numResultFunctionDicts)]
        resultValues = [[] for _ in range(numResultFunctionDicts)]
        
        for inputIdx, groupIdx in enumerate(clusters):
            for joinCol in joinCols:
                resultFunctionDicts[groupIdx][joinCol].append(functionDicts[inputIdx][joinCol])
            resultValues[groupIdx].append(values[inputIdx])
        resultValues = [x for x in resultValues if len(x) > 0]
        resultFunctionDicts = [{joinCol:pointwiseFunctionMax(np.array(x[joinCol])) for joinCol in joinCols} for x in resultFunctionDicts if len(x[joinCols[0]]) > 0] 
        return resultValues, resultFunctionDicts
            
    def decrementValue(self, value):
        if isinstance(value, str):
            return value
        else:
            return value - 2**-18
            
    def incrementValue(self, value):
        if isinstance(value, str):
            return value
        else:
            return value + 2**-18
        
    def getTriGramsFromString(self, str compValue):
        triGrams = [compValue[i:i+3] for i in range(0,len(compValue)-2)]
        return [triGram for triGram in triGrams if '%' not in triGram]
    
    def getMinFunctionsSatisfyingPredicates(self, listOfPreds, joinCols):
        if self.numValues == 0:
            return {joinCol: getEmptyFunction() for joinCol in joinCols}
        
        
        if self.notNullValues == 0:
            hasNotNullPreds = any([x.predType != "IS NULL" for x in listOfPreds])
            if hasNotNullPreds:
                return {joinCol: getEmptyFunction() for joinCol in joinCols}
            else:
                return {joinCol: x.copy() for joinCol, x in self.nullFunctions.items() if joinCol in joinCols}
        
        leftVal = self.min
        rightVal = self.max
        validFunctions = {x : [] for x in joinCols}
        inPreds = []
        for pred in listOfPreds:
            if pred.predType == "=":
                for joinCol in joinCols:
                    potentialEqualityFuncs = [self.equalityMaxFunctions[joinCol]]
                    comparisonValue = pred.compValue 
                    if not isinstance(pred.compValue, str):
                        comparisonValue = str(float(comparisonValue))
                    for i in range(len(self.bloomFilters)):
                        isInFilter = comparisonValue in self.bloomFilters[i]
                        if isInFilter:
                            potentialEqualityFuncs.append(self.representativeEqualityFunctionDicts[i][joinCol])
                    validFunctions[joinCol].append(pointwiseFunctionMax(np.array(potentialEqualityFuncs)))
            elif pred.predType == "<" or pred.predType == "<=":
                rightVal = min(rightVal, pred.compValue)
                if pred.predType == "<":
                    rightVal = self.decrementValue(rightVal)
            elif pred.predType == ">" or pred.predType == ">=":
                leftVal = max(leftVal, pred.compValue)
                if pred.predType == ">":
                    leftVal = self.incrementValue(leftVal)
            elif pred.predType == "IS NULL" and self.trackNulls:
                for joinCol in joinCols:
                    validFunctions[joinCol].append(self.nullFunctions[joinCol])
            elif pred.predType == "IS NOT NULL" and self.trackNulls:
                for joinCol in joinCols:
                    validFunctions[joinCol].append(self.notNullFunctions[joinCol])
            elif pred.predType == "LIKE" and self.trackTriGrams:
                queryTriGrams = self.getTriGramsFromString(pred.compValue)
                hasOutlierTriGram = False
                for triGram in queryTriGrams:
                    for i, triGramSet in enumerate(self.outlierTriGramGroupSets):
                        if triGram in triGramSet:
                            triGramFunctionDict = self.representativeTriGramFunctionDicts[i]
                            hasOutlierTriGram = True
                            for joinCol in joinCols:
                                validFunctions[joinCol].append(triGramFunctionDict[joinCol])
                if not hasOutlierTriGram:
                    triGramFunctionDict = self.triGramRemainderFunctionDict
                    for joinCol in joinCols:
                        validFunctions[joinCol].append(triGramFunctionDict[joinCol])

        if self.numBins > 0:
            qualifyingIntervals = []
            for interval in self.intervalToFunctionDictIdx.keys():
                if interval[0]<max(self.min, leftVal) and min(rightVal, self.max) <= interval[1]:
                    qualifyingIntervals.append(interval)
            for interval in qualifyingIntervals:
                functionDictIdx = self.intervalToFunctionDictIdx[interval]
                functionDict = self.representativeRangeFunctionDicts[functionDictIdx]
                for joinCol in joinCols:
                    validFunctions[joinCol].append(functionDict[joinCol].copy())
            if len(qualifyingIntervals) == 0:
                for joinCol in joinCols:
                    validFunctions[joinCol].append(getEmptyFunction())
        finalFunctions = dict()
        for joinCol in joinCols:
            finalFunctions[joinCol] = pointwiseFunctionMin(np.array(validFunctions[joinCol]))
        return finalFunctions
    
    def printDiagnostics(self):
        print("Filter Column Has " + str(self.numValues) + " Values")
        for i in range(len(self.intervals)):
            print("Interval: (" + str(self.intervals[i][0]) + ", " + str(self.intervals[i][1]) + "]")
            for joinCol in self.joinCols:
                print("Function for " + joinCol + ":")
                if self.intervals[i] in self.outlierFunctionsPerJoinCol[joinCol]:
                    print("[Outlier]")
                    self.outlierFunctionsPerJoinCol[joinCol][self.intervals[i]].printDiagnostics()
                else:
                    print("[Standard]")
                    self.maxFunctionsPerJoinColPerStepSize[joinCol][self.intervals[i][2]].printDiagnostics()
                    
    def memory(self):
        intervalFootprint = 0
        rangeFootprint = 0
        equalityBloomFilterFootprint = 0
        equalityOutlierFootprint = 0
        equalityMaxFootprint = 0
        triGramFootprint = 0
        nullFootprint = 0
        if self.trackNulls:
            for joinCol in self.joinCols:    
                nullFootprint += self.notNullFunctions[joinCol].memory() + self.nullFunctions[joinCol].memory()
        if self.notNullValues > 0:
            if self.numBins > 0 :
                for interval, functionDictIdx in self.intervalToFunctionDictIdx.items():
                    intervalFootprint += functionDictIdx.__sizeof__() + interval.__sizeof__()
                for joinCol in self.joinCols:
                    for funcDict in self.representativeRangeFunctionDicts:
                        rangeFootprint += funcDict[joinCol].memory()

            for bloomFilter in self.bloomFilters:
                equalityBloomFilterFootprint += bloomFilter.num_bits/8
            for funcDict in self.representativeEqualityFunctionDicts:
                for joinCol in self.joinCols:
                    equalityOutlierFootprint += funcDict[joinCol].memory()
            for joinCol in self.joinCols:
                equalityMaxFootprint += self.equalityMaxFunctions[joinCol].memory()

            if self.isStringColumn and self.trackTriGrams:
                for triGramSet in self.outlierTriGramGroupSets:
                    triGramFootprint += triGramSet.__sizeof__()
                for funcDict in self.representativeTriGramFunctionDicts:
                    for joinCol in self.joinCols:
                        triGramFootprint += funcDict[joinCol].memory()
                for joinCol in self.joinCols:
                    triGramFootprint += self.triGramRemainderFunctionDict[joinCol].memory()
                    
        print("Filter Col: " + self.filterCol)
        print("Interval Footprint: " + str(intervalFootprint))
        print("Range Footprint: " + str(rangeFootprint))
        print("Equality Bloom Filter Footprint: " + str(equalityBloomFilterFootprint))
        print("Equality Outlier Footprint: " + str(equalityOutlierFootprint))
        print("Equality Max Footprint: " + str(equalityMaxFootprint))
        print("TriGram Footprint: " + str(triGramFootprint))
        print("Null Footprint: " + str(nullFootprint))
        return intervalFootprint + rangeFootprint + equalityBloomFilterFootprint + equalityOutlierFootprint + equalityMaxFootprint + triGramFootprint + nullFootprint

        