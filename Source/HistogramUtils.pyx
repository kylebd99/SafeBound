# cython: infer_types=True
import pandas as pd
import numpy as np
from math import ceil, log2, sqrt
from bisect import bisect_left, bisect_right
from itertools import combinations
from collections import Counter
import concurrent.futures
import multiprocessing as mp
import string
import fastcluster as fc
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance_matrix
from pybloomfilter import BloomFilter
import sys, os
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory + 'Source')
from JoinGraphUtils import *
from PiecewiseConstantFunctionUtils cimport *
from PiecewiseConstantFunctionUtils import calculateBins


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


class MultiColumnFunctionHistogram:
    
    def __init__(self, table, filterCols, joinCols, numBins, numEqualityOutliers, relativeErrorPerSegment, numOutliers,
                 trackNulls, trackTriGrams, MPExecutor, groupingMethod, modelCDF, verbose):
        self.numRows = len(table)
        self.numBins = numBins
        self.numEqualityOutliers = numEqualityOutliers
        self.relativeErrorPerSegment = relativeErrorPerSegment
        self.numOutliers = numOutliers
        self.filterCols = filterCols.copy()
        self.joinCols = joinCols.copy()
        self.fullFunctions = dict()           
        self.hists = dict()
        self.trackNulls = trackNulls
        self.trackTriGrams = trackTriGrams
        if verbose > 0:
            print("Building Full Table Approximations")
        for joinCol in self.joinCols:
            valCounts = np.array(sorted(table[joinCol].value_counts(ascending=False).to_list(), reverse=True), dtype='int')
            function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment/4.0, modelCDF=modelCDF) # The number of segments is increased for the full approximation. 
            function.compressFunc()
            self.fullFunctions[joinCol] = function
        
        if len(filterCols) == 0:
            return
                
        self.equalityPairNames = []
        self.equalityPairCols = [[x[0], x[1]] if x[0] < x[1] else [x[1], x[0]] for x in combinations(self.filterCols, 2)]
        for col1, col2 in self.equalityPairCols:
            continue
            pairColName = col1 + "_String" + "_" + col2 + "_String"
            self.equalityPairNames.append(pairColName)
        
        
        if MPExecutor: 
            filterColArgs = []
            filterColNames = []
            for filterCol in self.filterCols:
                if verbose > 0:
                    print("Building Histogram: " + filterCol)
                filterColArgs.append([table[list(set([filterCol]+self.joinCols))], 
                                                  filterCol,
                                                  [],
                                                  self.joinCols.copy(),
                                                  self.numBins,
                                                  self.numEqualityOutliers, 
                                                  self.relativeErrorPerSegment,
                                                  self.numOutliers,
                                                  self.trackNulls,
                                                  self.trackTriGrams,
                                                  groupingMethod,
                                                  modelCDF,
                                                  verbose])
                filterColNames.append(filterCol)
            for filterCol, pairwiseFilterCols in zip(self.equalityPairNames, self.equalityPairCols):
                if verbose > 0:
                    print("Building Histogram: " + filterCol)
                filterColArgs.append([table[list(set(pairwiseFilterCols+self.joinCols))], 
                                                  filterCol,
                                                  pairwiseFilterCols,
                                                  self.joinCols.copy(),
                                                  0,
                                                  max(1, int(self.numEqualityOutliers/float(len(self.filterCols)))), 
                                                  self.relativeErrorPerSegment,
                                                  max(1, int(self.numOutliers/float(len(self.filterCols)))),
                                                  False,
                                                  False,
                                                  groupingMethod,
                                                  modelCDF,
                                                  verbose])
                filterColNames.append(filterCol)
            histograms = MPExecutor.starmap(VWFunctionHistogram, filterColArgs)
            for i in range(len(filterColNames)):
                self.hists[filterColNames[i]] = histograms[i]
        else:
            self.hists = dict()
            for filterCol in self.filterCols:
                if verbose > 0:
                    print("Building Histogram: " + filterCol)
                self.hists[filterCol]= VWFunctionHistogram(table=table,
                                                           filterCol = filterCol, 
                                                           pairwiseFilterCols = [],
                                                           joinCols = self.joinCols,
                                                           numBins = self.numBins, 
                                                           numEqualityOutliers = self.numEqualityOutliers,
                                                           relativeErrorPerSegment = self.relativeErrorPerSegment,
                                                           numOutliers = self.numOutliers,
                                                           trackNulls = trackNulls,
                                                           trackTriGrams = trackTriGrams,
                                                           groupingMethod = groupingMethod,
                                                           modelCDF=modelCDF,
                                                           verbose=verbose)
            for filterCol, pairwiseFilterCols in zip(self.equalityPairNames, self.equalityPairCols):
                if verbose>0:
                    print("Building Histogram: " + filterCol)
                self.hists[filterCol]= VWFunctionHistogram(table=table,
                                                           filterCol = filterCol, 
                                                           pairwiseFilterCols = pairwiseFilterCols,
                                                           joinCols = self.joinCols,
                                                           numBins = 0, 
                                                           numEqualityOutliers = max(1, int(self.numEqualityOutliers/float(len(self.filterCols)))),
                                                           relativeErrorPerSegment = self.relativeErrorPerSegment,
                                                           numOutliers = max(1, int(self.numOutliers/float(len(self.filterCols)))),
                                                           trackNulls = False,
                                                           trackTriGrams = False,
                                                           groupingMethod = groupingMethod,
                                                           modelCDF = modelCDF,
                                                           verbose=verbose)
    
        
    def getMinFunctionsSatisfyingPredicates(self, listOfPreds, joinCols):
        if self.numRows == 0:
            return {joinCol: getEmptyFunction() for joinCol in joinCols}
        numValidPreds = 0
        predsByColumn = dict()
        for pred in listOfPreds:
            predsByColumn[pred.colName] = []
        for pred in listOfPreds:
            if pred.predType in [">",">=","<=","<"] and self.hists[pred.colName].numBins == 0:
                continue
            elif pred.predType == "LIKE" and self.hists[pred.colName].trackTriGrams == False:
                continue
            elif pred.predType in ["IS NULL", "IS NOT NULL"]  and self.hists[pred.colName].trackNulls == False:
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
            joinColFunctions = self.hists[filterCol].getMinFunctionsSatisfyingPredicates(preds, joinCols)
            for i in range(len(joinCols)):
                joinCol = joinCols[i]
                function = joinColFunctions[joinCol]
                validFunctions[i].append(function)
        
        equalityPreds = [x for x in listOfPreds if x.predType == "="]
        if len(equalityPreds) > 1:
            pairsOfPreds = [(x, y) if x.colName < y.colName else (y,x) for x,y in combinations(equalityPreds, 2)]
            for pred1, pred2 in pairsOfPreds:
                pairwiseColName = pred1.colName + "_String" + "_" + pred2.colName + "_String"
                if pairwiseColName in self.equalityPairNames:
                    pred1CompValue = pred1.compValue
                    if not self.hists[pred1.colName].isStringColumn:
                        pred1CompValue = str(float(pred1CompValue))
                    pred2CompValue = pred2.compValue
                    if not self.hists[pred2.colName].isStringColumn:
                        pred2CompValue = str(float(pred2CompValue))
                    predValue = pred1CompValue + "_" + pred2CompValue
                    pairwisePred = Predicate(pred1.colName + "_" + pred2.colName, "=", predValue)
                    joinColFunctions = self.hists[pairwiseColName].getMinFunctionsSatisfyingPredicates([pairwisePred], joinCols)
                    for i in range(len(joinCols)):
                        joinCol = joinCols[i]
                        function = joinColFunctions[joinCol]
                        validFunctions[i].append(function)
    
        minFunctionsDict = dict()
        for i in range(len(joinCols)):
            joinCol = joinCols[i]
            minFunctionsDict[joinCol] = pointwiseFunctionMin(np.array(validFunctions[i]))
        return minFunctionsDict
        
    def printHists(self):
        for col in self.filterCols:
            print("Histogram For Column: " + col)
            self.hists[col].printHist()

    def memory(self):
        footprint = 0
        for col, hist in self.hists.items():
            footprint += hist.memory()
        for joinCol in self.joinCols:
            footprint += self.fullFunctions[joinCol].memory()
        return footprint
    
# This histogram is represented as a series of intervals (a_i,b_i] which are each associated with a particular function approximation f_i.
# When queries are evaluated, the smallest interval which captures the predicate has its associated function returned.
class VWFunctionHistogram:

    def __init__(self, table, filterCol, pairwiseFilterCols, joinCols, numBins, numEqualityOutliers, relativeErrorPerSegment, 
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
        
        
        if len(pairwiseFilterCols) == 2:
            self.equalityMaxFunctions = dict()
            notNullTable = table.loc[:,list(set(pairwiseFilterCols + joinCols))]
            notNullTable = notNullTable.loc[(~notNullTable[pairwiseFilterCols[0]].isna()) & (~notNullTable[pairwiseFilterCols[1]].isna())]
            self.notNullValues = len(notNullTable)
            if self.notNullValues == 0:
                return
            notNullTable[filterCol] = notNullTable[pairwiseFilterCols].apply(tuple, axis=1)    
            if not pairwiseFilterCols[0] in joinCols: 
                notNullTable.drop([pairwiseFilterCols[0]], axis=1, inplace=True)
            if not pairwiseFilterCols[1] in joinCols: 
                notNullTable.drop([pairwiseFilterCols[1]], axis=1, inplace=True)
            self.min = min(notNullTable[filterCol])
            self.max = max(notNullTable[filterCol])
            self.isStringColumn = True
            
            notNullTable.sort_values(filterCol, inplace=True)
            numFilterValues = notNullTable[filterCol].nunique()
            
            filterOutlierValCounts = notNullTable[filterCol].value_counts().head(self.numEqualityOutliers).reset_index()
            outlierVals = filterOutlierValCounts["index"]
            outlierCounts = filterOutlierValCounts[filterCol]
            
            outlierFunctionDicts = []
            for curVal in outlierVals:
                leftIdx = bisect_left(notNullTable[filterCol].array, curVal)
                rightIdx = bisect_right(notNullTable[filterCol].array, curVal)
                score = 0
                functionDict = dict()
                for joinCol in joinCols:
                    valCounts = np.array(sorted(notNullTable[joinCol].iloc[leftIdx:rightIdx].value_counts(ascending=False).to_list(), 
                                                reverse=True), 
                                         dtype='int')
                    function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                    function.compressFunc()
                    functionDict[joinCol] = function
                outlierFunctionDicts.append(functionDict)

            self.bloomFilters = []
            groupValues, self.representativeEqualityFunctionDicts = VWFunctionHistogram.coalesceFunctionDicts(outlierVals,
                                                                                                              outlierFunctionDicts, 
                                                                                                              joinCols,
                                                                                                              self.numOutliers,
                                                                                                              groupingMethod,
                                                                                                              modelCDF)
            for i in range(len(groupValues)):
                vals = groupValues[i]
                bloomFilter = BloomFilter(len(vals), .00001)
                for val in vals:
                    lval = val[0]
                    rval = val[1]
                    if not isinstance(lval, str): 
                        lval = str(float(lval))
                    if not isinstance(rval, str): 
                        rval = str(float(rval))
                    bloomFilter.add(lval + "_" + rval)
                self.bloomFilters.append(bloomFilter)

            # To handle the remaining equality values, Groupby into [filterCol, size], 
            # then rank these pairs by size within each filterCol value. Then,take the maximum over each rank. 
            # This gives you the pointwise maximum degree function. Lastly, truncate it to the maximum number of
            # rows per filter value.
            remainderTable = notNullTable[~notNullTable[filterCol].isin(outlierVals)]
            remainderMaxCount = remainderTable.groupby(filterCol).size().max()
            for joinCol in joinCols:
                if len(remainderTable) == 0:
                    self.equalityMaxFunctions[joinCol] = getEmptyFunction()
                    continue

                remainderMaxDegrees = None
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
            return

                
        self.isStringColumn = table[filterCol].dtype == 'object'
        notNullTable = table.loc[~table[filterCol].isna()]
        notNullTable.sort_values(filterCol, inplace=True)
        if self.trackNulls:
            if verbose > 0:
                print("Approximating null distributions: "+ filterCol)
            # Record the distributions of the join columns for Is Null and Is Not Null Predicates.
            nullTable = table.loc[table[filterCol].isna()]
            self.nullFunctions = dict()
            for joinCol in joinCols:
                valCounts = np.array(sorted(nullTable[joinCol].value_counts(ascending=False).to_list(), reverse=True), dtype='int')
                if len(valCounts) == 0:
                    self.nullFunctions[joinCol] = getEmptyFunction()
                else:
                    function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                    function.compressFunc()
                    self.nullFunctions[joinCol] = function

            if verbose > 0:
                print("Approximating not null distributions: "+ filterCol)
            self.notNullFunctions = dict()
            for joinCol in joinCols:
                valCounts = np.array(sorted(notNullTable[joinCol].value_counts(ascending=False).to_list(), reverse=True), dtype='int')
                if len(valCounts) == 0:
                    self.notNullFunctions[joinCol] = getEmptyFunction()
                else:
                    function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                    function.compressFunc()
                    self.notNullFunctions[joinCol] = function
        
        
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
                print("Approximating range distributions: "+ filterCol)
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
                print("Generating Range Functions: "+ filterCol)
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
            groupIntervals, self.representativeRangeFunctionDicts = VWFunctionHistogram.coalesceFunctionDicts(nonEdgeIntervals, 
                                                                                                              nonEdgeIntervalFunctionDicts, 
                                                                                                              joinCols,
                                                                                                              self.numOutliers,
                                                                                                              groupingMethod,
                                                                                                              modelCDF)
            for i in range(len(edgeIntervals)):
                groupIntervals.append([edgeIntervals[i]])
                self.representativeRangeFunctionDicts.append(edgeIntervalDicts[i])

            self.intervalToFunctionDictIdx = dict()
            for i in range(len(groupIntervals)):
                for interval in groupIntervals[i]:
                    self.intervalToFunctionDictIdx[interval] = i
            del self.intervals
        
        # In order to not calculate a function for every filter value X join col pair, we are instead going to do some approximation.
        # First, we take a rough selection of outlier distributions based on the row-count of the equality predicates. For high row-count
        # predicates, we calculate their function approximations then do a similar grouping as in the intervals above.
        # We represent each grouping via a bloom filter to save space and speed querying. The remainder are all grouped and maxed as
        # one function using batch methods in pandas.
        if verbose > 0:
            print("Approximating equality distributions: "+ filterCol)
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
                valCounts = np.array(sorted(notNullTable[joinCol].iloc[leftIdx:rightIdx].value_counts(ascending=False).to_list(), 
                                            reverse=True), 
                                     dtype='int')
                function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                function.compressFunc()
                functionDict[joinCol] = function
            outlierFunctionDicts.append(functionDict)
        
        self.bloomFilters = []
        groupValues, self.representativeEqualityFunctionDicts = VWFunctionHistogram.coalesceFunctionDicts(outlierVals,
                                                                                                          outlierFunctionDicts, 
                                                                                                          joinCols,
                                                                                                          self.numOutliers,
                                                                                                          groupingMethod,
                                                                                                          modelCDF)
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
        for joinCol in joinCols:
            if len(remainderTable) == 0:
                self.equalityMaxFunctions[joinCol] = getEmptyFunction()
                continue
                
            remainderMaxDegrees = None
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
                
        # We handle string columns separately because they may have LIKE predicates applied to them.
        if self.trackTriGrams and self.isStringColumn:
            
            if verbose > 0:
                print("Handling Most Common TriGrams: "+ filterCol + " " + str(len(notNullTable)))
                
            triGramCol = filterCol + "_TriGrams"
            triGramCounts = Counter([triGram for x in notNullTable[filterCol].sample(min(len(notNullTable)-1, 100000)) for triGram in self.getTriGramsFromString(x)])
            mostCommonTriGrams = [x[0] for x in triGramCounts.most_common(self.numEqualityOutliers)]
            compressedFilterColumn = pd.DataFrame(notNullTable[filterCol].unique(), columns=[filterCol]).set_index(filterCol)
            mostCommonTriGramIndices = {x:[] for x in mostCommonTriGrams}
            for i, val in enumerate(notNullTable[filterCol]):
                for triGram in self.getTriGramsFromString(val):
                    if triGram in mostCommonTriGramIndices:
                        mostCommonTriGramIndices[triGram].append(i)
            
            outlierFunctionDicts = []
            for triGram in mostCommonTriGrams:
                subTable = notNullTable.iloc[mostCommonTriGramIndices[triGram]]
                functionDict = dict()
                for joinCol in joinCols:
                    valCounts = np.array(subTable[joinCol].value_counts(ascending=False).to_list(), 
                                         dtype='int')
                    function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                    function.compressFunc()
                    functionDict[joinCol] = function
                outlierFunctionDicts.append(functionDict)
            groupValues, self.representativeTriGramFunctionDicts = VWFunctionHistogram.coalesceFunctionDicts(mostCommonTriGrams,
                                                                                                              outlierFunctionDicts, 
                                                                                                              joinCols,
                                                                                                              self.numOutliers,
                                                                                                              groupingMethod,
                                                                                                              modelCDF)
            
            self.outlierTriGramGroupDict = dict()
            for i, triGrams in enumerate(groupValues):
                for triGram in triGrams:
                    self.outlierTriGramGroupDict[triGram] = i
            
            if verbose > 0:
                print("Handling Remainder Rows: "+ filterCol + " " + str(len(notNullTable)))
                
            # If no outlier TriGrams are present in the query string, we return the degree sequence of all rows
            # which contain none of the outlier TriGrams
            mostCommonTriGramsSet = set(mostCommonTriGrams)
            remainderTable = notNullTable[notNullTable[filterCol].apply(lambda x: \
                                                                         all([y not in mostCommonTriGramsSet for y in \
                                                                              self.getTriGramsFromString(x)]))]
            self.triGramRemainderFunctionDict = dict()
            for joinCol in joinCols:
                valCounts = np.array(sorted(remainderTable[joinCol]\
                                            .value_counts(ascending=False).to_list(), 
                                            reverse=True), 
                                     dtype='int')
                function = PiecewiseConstantFunction(valCounts, self.relativeErrorPerSegment, modelCDF=modelCDF)
                function.compressFunc()
                self.triGramRemainderFunctionDict[joinCol] = function
            
        if verbose > 0:
            print("Finished Building Histogram: " + filterCol)
    
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
                if modelCDF == True:
                    resultFunctionDicts.append({joinCol: pointwiseFunctionMax(np.array([x[joinCol] for x in sortedFunctionDicts[left:right]])) for joinCol in joinCols})
                elif modelCDF == False:
                    resultFunctionDicts.append({joinCol: pointwisePDFMax(np.array([x[joinCol] for x in sortedFunctionDicts[left:right]])) for joinCol in joinCols})
            return resultValues, resultFunctionDicts
            
        
        joinColToSegments = dict()
        for joinCol in joinCols:
            maxNumValues = max([functionDict[joinCol].rightIntervalEdges[-1] for functionDict in functionDicts])
            maxNumRows = max([functionDict[joinCol].getNumRows() for functionDict in functionDicts])
            maxSelfJoinSize = max([functionDict[joinCol].getSelfJoinSize() for functionDict in functionDicts])
            maxDegree = max([functionDict[joinCol].constants[0] for functionDict in functionDicts])
            joinColToSegments[joinCol] = calculateBinsLinear(maxNumValues, 200)
        
        
        functionDictVectors = []
        for functionDict in functionDicts:
            functionDictVector = np.array([], dtype='float')
            for joinCol in joinCols:
                segments = joinColToSegments[joinCol]
                function = functionDict[joinCol]
                numRows = max(1, function.getNumRows())
                # cdfShape = [function.calculateRowsAtPoint(x)/numRows for x in segments]
                # functionDictVector = np.append(functionDictVector, cdfShape)
                totalRows =  numRows / maxNumRows * len(segments) * 3
                totalSelfJoinSize = function.getSelfJoinSize() / maxSelfJoinSize * len(segments) * 3
                totalMaxDegree = function.constants[0] / maxDegree * len(segments)
                functionDictVector = np.append(functionDictVector, [totalRows, totalSelfJoinSize, totalMaxDegree])
            functionDictVectors.append(functionDictVector)
        functionDictVectors = np.array(functionDictVectors, dtype=np.float64)
        
        dendrogram = None
        if groupingMethod == "CompleteClustering":
            dendrogram =  fc.linkage(functionDictVectors, method='complete', metric='canberra')
        elif groupingMethod == "SingleClustering":
            dendrogram =  fc.linkage(functionDictVectors, method='single', metric='canberra')
        clusters = sch.fcluster(dendrogram, numResultFunctionDicts, criterion='maxclust') - 1 # Subtract 1 so that group labels match indices 
        resultFunctionDicts = [{joinCol:[] for joinCol in joinCols} for _ in range(numResultFunctionDicts)]
        resultValues = [[] for _ in range(numResultFunctionDicts)]
        
        for inputIdx, groupIdx in enumerate(clusters):
            for joinCol in joinCols:
                resultFunctionDicts[groupIdx][joinCol].append(functionDicts[inputIdx][joinCol])
            resultValues[groupIdx].append(values[inputIdx])
        resultValues = [x for x in resultValues if len(x) > 0]
        if modelCDF == True:
            resultFunctionDicts = [{joinCol:pointwiseFunctionMax(np.array(x[joinCol])) for joinCol in joinCols} for x in resultFunctionDicts if len(x[joinCols[0]]) > 0] 
        elif modelCDF == False:
            resultFunctionDicts = [{joinCol:pointwisePDFMax(np.array(x[joinCol])) for joinCol in joinCols} for x in resultFunctionDicts if len(x[joinCols[0]]) > 0] 
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
                    triGramFunctionDict = None
                    if triGram in self.outlierTriGramGroupDict:
                        triGramFunctionDict = self.representativeTriGramFunctionDicts[self.outlierTriGramGroupDict[triGram]]
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
    
    def printHist(self):
        print("Histogram Holds " + str(self.numValues) + " Values")
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
                triGramFootprint += self.outlierTriGramGroupDict.__sizeof__()
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

        
