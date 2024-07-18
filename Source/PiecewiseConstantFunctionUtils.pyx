# cython: infer_types=True,boundscheck=False
from math import floor, ceil, log, sqrt, prod
from bisect import bisect_right
import numpy as np

cpdef PiecewiseConstantFunction getEmptyFunction():
    return PiecewiseConstantFunction(np.array([], dtype='int'), 0)

cpdef long bisect_left_double(double[:] a, double x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if a[mid] < x: lo = mid + 1
        else: hi = mid
    return lo

cpdef long bisect_left_long(long[:] a, double x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if a[mid] < x: lo = mid + 1
        else: hi = mid
    return lo

def calculateBins(long numVals, long numSegments):
    cdef long[:] bins = np.zeros(0, dtype='int')
    if numVals > numSegments*10:
        bins = np.concatenate((np.linspace(1, int(numSegments/4), int(numSegments/4), dtype="int"), 
                               np.geomspace(int(numSegments/4), int(numSegments), int(max(2, numSegments/4)), dtype="int", endpoint=True), 
                               ), 0, dtype='int')
    elif numVals <= numSegments:
        bins = np.linspace(1, int(numVals), int(numSegments), dtype='int', endpoint=True)
    else:
        bins = np.geomspace(1, int(max(1, numVals)), int(max(2, min(numSegments, numVals/5))), endpoint=True, dtype="int")
    return bins

cpdef long[:] calculateBinsExponential(long numVals, long numSegments):
    cdef long[:] bins = np.geomspace(1, int(max(1, numVals)), numSegments, dtype="int", endpoint=True)
    return bins

cpdef long[:] calculateBinsLinear(long numVals, long numSegments):
    cdef long[:] bins = np.linspace(1, int(max(1, numVals)), numSegments, dtype="int", endpoint=True)
    return bins

cpdef long[:] calculateBinsEquiDepth(long[:] data, long numSegments):
    cardinality = sum(data)
    cardinalityPerSegment = cardinality/numSegments
    bins = []
    curSegmentCardinality = 0
    for i, value in enumerate(data):
        if curSegmentCardinality > cardinalityPerSegment:
            bins.append(i)
            curSegmentCardinality = 0
        curSegmentCardinality += value
    return np.array(bins, dtype='long')

cpdef long[:] calculateBinsRelativeError(long[:] data, double relativeError):
    cdef long[:] rightEdges = np.zeros(100, dtype='long')
    cdef long currentConstant = data[0]
    cdef double currentApproxValue = 0
    cdef double currentTrueValue = 0
    cdef double curRight = 0
    cdef double curRows = 0
    cdef double totalValue = sum([x**2 for x in data])
    cdef int numEdges = 0
    for i, val in enumerate(data):
        currentApproxValue += data[min(len(data)-1, floor(curRight))]*val
        currentTrueValue += val**2
        curRight += float(val)/currentConstant
        if currentApproxValue - currentTrueValue > relativeError * totalValue:
            currentApproxValue = 0
            currentTrueValue = 0
            rightEdges[numEdges] = i
            currentConstant = val
            numEdges += 1
        if numEdges >= len(rightEdges)-1:
            break
    rightEdges[numEdges] = len(data)
    numEdges += 1
    return rightEdges[0:numEdges]

cdef PiecewiseConstantFunction pointwiseFunctionMax(PiecewiseConstantFunction[:] functions):
    if len(functions) == 0:
        return getEmptyFunction()
    elif len(functions) == 1:
        return functions[0]
    
    cdef long[:] indices = np.zeros(len(functions), dtype='int')
    functionHasSpace = [True for _ in functions]
    left = 0
    right = 0
    constants = []
    segmentLengths = []
    cumulativeRowsPerFunction = [0 for _ in functions]
    prevCumulativeRows = 0
    maxRowCount = max([x.getNumRows() for x in functions])
    maxSegmentCount = max([len(x.rightIntervalEdges) for x in functions])
    cdef PiecewiseConstantFunction curFunction = functions[0] # Need to set the type of curFunction early to avoid compiler errors
    while any(functionHasSpace):
        right = min([functions[i].rightIntervalEdges[indices[i]] for i in range(len(functions)) if functionHasSpace[i]])
        if right > left:
            curCumulativeRows = 0
            for i in range(len(functions)):
                curFunction = functions[i] 
                if functionHasSpace[i]:
                    curCumulativeRows = max(curFunction.calculateRowsAtPoint(right), curCumulativeRows)
                else:
                    curCumulativeRows = max(curFunction.cumulativeRows[-1], curCumulativeRows)
            
            constant = (curCumulativeRows - prevCumulativeRows)/(right-left)
            if constant > 0:
                constants.append(constant)
                segmentLengths.append(right-left)
                left = right
                prevCumulativeRows =  curCumulativeRows
            
            if curCumulativeRows >= maxRowCount:
                break
        
        for i in range(len(functions)):
            if functionHasSpace[i] and functions[i].rightIntervalEdges[indices[i]] <= right:
                indices[i] += 1
        functionHasSpace = [indices[i] < len(functions[i].rightIntervalEdges) for i in range(len(functions))]
    constants, segmentLengths = zip(*sorted(zip(constants, segmentLengths), reverse=True))
    cumulativeRows = []
    rightEdges = []
    curRow = 0
    curEdge = 0
    for i in range(len(constants)):
        curRow += constants[i] * segmentLengths[i]
        curEdge += segmentLengths[i]
        cumulativeRows.append(curRow)
        rightEdges.append(curEdge)
    func = getEmptyFunction()
    func.rightIntervalEdges = np.array(rightEdges, dtype='double')
    func.constants = np.array(constants, dtype='double')
    func.cumulativeRows = np.array(cumulativeRows, dtype='double')
    func.compressFunc(maxSegmentCount)
    return func

cpdef double pairwiseMaxAndSelfJoin(double[:] rightEdges_L, 
                                   double[:] rightEdges_R, 
                                   double[:] cumulativeRows_L, 
                                   double[:] cumulativeRows_R,
                                   int len_L,
                                   int len_R) nogil:
    cdef double maxSelfJoin = 0
    cdef double prevCumulativeRows = 0
    cdef double prevRightEdge = 0
    cdef double curCumulativeRows = 0
    cdef double curRightEdge = 0
    cdef double curConstant = 0
    cdef double curSegmentLength = 0
    cdef int idx_L = 0
    cdef int idx_R = 0
    cdef int finished_L = 0
    cdef int finished_R = 0
    
    while finished_L == 0 or finished_R == 0:
        if finished_L == 0 and finished_R == 0:
            curCumulativeRows = max(cumulativeRows_L[idx_L], cumulativeRows_R[idx_R])
            curRightEdge = max(rightEdges_L[idx_L], rightEdges_R[idx_R])
        elif finished_L == 0:
            curCumulativeRows = cumulativeRows_L[idx_L]
            curRightEdge = rightEdges_L[idx_L]
        elif finished_R == 0:
            curCumulativeRows = cumulativeRows_R[idx_R]
            curRightEdge = rightEdges_R[idx_R]
        curSegmentLength = (curRightEdge-prevRightEdge)
        if curSegmentLength > 0:
            curConstant = (curCumulativeRows - prevCumulativeRows)/curSegmentLength
            maxSelfJoin = maxSelfJoin + curConstant*curConstant*curSegmentLength
        while finished_L == 0 and curRightEdge >= rightEdges_L[idx_L]:
            idx_L = idx_L + 1
            if idx_L >= len_L:
                finished_L = 1
        while finished_R == 0 and curRightEdge >= rightEdges_R[idx_R]:
            idx_R = idx_R + 1
            if idx_R >= len_R:
                finished_R = 1
        prevCumulativeRows = curCumulativeRows
        prevRightEdge = curRightEdge

    return maxSelfJoin
    

cpdef PiecewiseConstantFunction pointwisePDFMax(PiecewiseConstantFunction[:] functions):
    if len(functions) == 0:
        return getEmptyFunction()
    elif len(functions) == 1:
        return functions[0]
    
    cdef long[:] indices = np.zeros(len(functions), dtype='int')
    maxSegmentCount = max([len(x.rightIntervalEdges) for x in functions])
    functionHasSpace = [True for _ in functions]
    left = 0
    right = 0
    constants = []
    rightIntervalEdges = []
    cumulativeRows = []
    curRow = 0
    while any(functionHasSpace):
        right = min([functions[i].rightIntervalEdges[indices[i]] for i in range(len(functions)) if functionHasSpace[i]])
        if right > left:
            curDegree = max([functions[i].constants[indices[i]] for i in range(len(functions)) if functionHasSpace[i]])
            curRow += (right-left)*curDegree
            constants.append(curDegree)
            rightIntervalEdges.append(right)
            cumulativeRows.append(curRow)
            left = right
        for i in range(len(functions)):
            if functionHasSpace[i] and functions[i].rightIntervalEdges[indices[i]] <= right:
                indices[i] += 1
        functionHasSpace = [indices[i] < len(functions[i].rightIntervalEdges) for i in range(len(functions))]
    
    func = getEmptyFunction()
    func.rightIntervalEdges = np.array(rightIntervalEdges, dtype='double')
    func.constants = np.array(constants, dtype='double')
    func.cumulativeRows = np.array(cumulativeRows, dtype='double')
    func.compressFunc(maxSegmentCount)
    return func

cpdef PiecewiseConstantFunction pointwiseFunctionMin(PiecewiseConstantFunction[:] functions):
    if len(functions) == 0:
        return getEmptyFunction()
    elif len(functions) == 1:
        return functions[0]
    cdef long[:] indices = np.zeros(len(functions), dtype='int')
    functionHasSpace = [True for _ in functions]
    left = 0
    right = 0
    rightEdges = []
    constants = []
    cumulativeRows = []
    cumulativeRowsPerFunction = [0 for _ in functions]
    prevCumulativeRows = 0
    minRowCount = min([x.getNumRows() for x in functions])
    while any(functionHasSpace):
        right = min([functions[i].rightIntervalEdges[indices[i]] for i in range(len(functions)) if functionHasSpace[i]])        
        if right > left:
            cumulativeRowsPerFunction = [cumulativeRowsPerFunction[i] + (right-left)*functions[i].constants[indices[i]]  if functionHasSpace[i] else cumulativeRowsPerFunction[i] for i in range(len(functions))]
            curCumulativeRows = min(cumulativeRowsPerFunction)
            constant = (curCumulativeRows - prevCumulativeRows)/(right-left)
            if constant == 0:
                constant = .0001
            rightEdges.append(right)
            cumulativeRows.append(curCumulativeRows)
            constants.append(constant)
            left = right
            prevCumulativeRows =  curCumulativeRows
            if curCumulativeRows == minRowCount:
                break

        for i in range(len(functions)):
            if functionHasSpace[i] and functions[i].rightIntervalEdges[indices[i]] <= right:
                indices[i] += 1
        functionHasSpace = [indices[i] < len(functions[i].rightIntervalEdges) for i in range(len(functions))]
    func = getEmptyFunction()
    func.rightIntervalEdges = np.array(rightEdges, dtype='double')
    func.constants = np.array(constants, dtype='double')
    func.cumulativeRows = np.array(cumulativeRows, dtype='double')
    func.compressFunc()
    return func


cpdef PiecewiseConstantFunction pointwiseFunctionMult(PiecewiseConstantFunction[:] functions):
    if len(functions) == 0:
        return getEmptyFunction()
    elif len(functions) == 1:
        return functions[0]
    cdef long[:] indices = np.zeros(len(functions), dtype='int')
    functionHasSpace = [True for _ in functions]
    left = 0
    right = 0
    curCumulativeRows = 0
    rightEdges = []
    constants = []
    cumulativeRows = []
    while all(functionHasSpace):
        right = min([functions[i].rightIntervalEdges[indices[i]] for i in range(len(functions)) if functionHasSpace[i]])
        constant =  prod([functions[i].constants[indices[i]] for i in range(len(functions)) if functionHasSpace[i]])
        curCumulativeRows += (right - left) * constant
        if constant == 0:
            constant = .0001
        rightEdges.append(right)
        constants.append(constant)
        cumulativeRows.append(curCumulativeRows)
        for i in range(len(functions)):
            if functionHasSpace[i] and functions[i].rightIntervalEdges[indices[i]] <= right:
                indices[i] += 1
        functionHasSpace = [indices[i] < len(functions[i].rightIntervalEdges) for i in range(len(functions))]
        left = right
    
    func = getEmptyFunction()
    func.rightIntervalEdges = np.array(rightEdges, dtype='double')
    func.constants = np.array(constants, dtype='double')
    func.cumulativeRows = np.array(cumulativeRows, dtype='double')
    func.compressFunc()
    return func

cpdef inline double estimatedFunctionWeight(PiecewiseConstantFunction function):
#    functionWeight =  0
#    left = 0
#    for i in range(len(function.constants)):
#        right = function.rightIntervalEdges[i]
#        functionWeight += function.constants[i]*(right-left)
#    return functionWeight
    return function.getNumRows()

# The difference score is roughly equal to the relative error imposed on queries that refer to each predicate.
cpdef double pairwiseFunctionDifference(PiecewiseConstantFunction function1, PiecewiseConstantFunction function2):
    index1 = 0
    index2 = 0
    function1HasSpace = True
    function2HasSpace = True
    left = 0
    right = 0
    differenceSum = 0
    while function1HasSpace or function2HasSpace:
        constantDifference = 0
        if function1HasSpace and function2HasSpace:
            right = min(function1.rightIntervalEdges[index1], function1.rightIntervalEdges[index1])
            constantDifference = abs(function1.constants[index1]-function2.constants[index2])
        elif function1HasSpace:
            right = function1.rightIntervalEdges[index1]
            constantDifference = function1.constants[index1]
        else:
            right = function2.rightIntervalEdges[index2]
            constantDifference = function2.constants[index2]
        differenceSum += constantDifference*(right-left)
        if function1HasSpace and function1.rightIntervalEdges[index1] == right:
            index1 += 1
        if function2HasSpace and function2.rightIntervalEdges[index2] == right:
            index2 += 1
        function1HasSpace = index1 < len(function1.rightIntervalEdges)
        function2HasSpace = index2 < len(function2.rightIntervalEdges)
        left = right
    
    function1Weight = estimatedFunctionWeight(function1)
    function2Weight = estimatedFunctionWeight(function2)

    differenceSum = float(differenceSum/function1Weight + differenceSum/function2Weight) 
    return differenceSum

cpdef PiecewiseConstantFunction arraysToPiecewiseConstantFunction(rightIntervalEdges, cumulativeRows, constants):
    newFunc = getEmptyFunction()
    newFunc.setRightIntervalEdges(np.array(rightIntervalEdges, dtype='double'))
    newFunc.setCumulativeRows(np.array(cumulativeRows, dtype='double'))
    newFunc.setConstants(np.array(constants, dtype='double'))
    return newFunc

cdef class PiecewiseConstantFunction:
    
    def __init__(self, long[:] data, double relativeErrorPerSegment, str segmentationStrategy = "RelativeError", bint modelCDF = True):
        if data is None or len(data) == 0:
            self.rightIntervalEdges = np.array([1.])
            self.cumulativeRows = np.array([1.])
            self.constants = np.array([.0001])
            return
        
        cdef double totalRows = sum(data)
        cdef long[:] binRightEdges = np.array([], dtype=np.int64)
        
        if segmentationStrategy == "Exponential":
            binRightEdges = calculateBinsExponential(len(data), int(relativeErrorPerSegment))
        elif segmentationStrategy == "Linear":
            binRightEdges = calculateBinsLinear(len(data), int(relativeErrorPerSegment))
        elif segmentationStrategy == "EquiDepth":
            binRightEdges = calculateBinsEquiDepth(data, int(relativeErrorPerSegment))
        elif segmentationStrategy == "RelativeError":
            binRightEdges = calculateBinsRelativeError(data, relativeErrorPerSegment)
        
        for i in range(len(binRightEdges)):
            binRightEdges[i] = len(data) - bisect_left_long(data[::-1], data[binRightEdges[i]-1])
        binRightEdges = np.array(sorted(set(binRightEdges)), dtype='int')
        
        self.rightIntervalEdges = np.zeros(len(binRightEdges), dtype='double')
        self.cumulativeRows = np.zeros(len(binRightEdges), dtype='double')
        self.constants = np.zeros(len(binRightEdges), dtype='double')
        cdef long counter = 0
        cdef long curLeftData = 0
        cdef long curRightData = 0
        cdef double curRightEdge = 0
        cdef double curRow = 0
        for i in range(len(binRightEdges)):
            curRightData = binRightEdges[i]
            if curLeftData == curRightData:
                continue
            if modelCDF == True:
                if curRow +(curRightData-curLeftData)*data[curLeftData] >= totalRows:
                    curRight = curRightEdge + ceil((totalRows-curRow)/data[curLeftData])
                    self.rightIntervalEdges[counter] = curRight
                    self.cumulativeRows[counter] = totalRows
                    self.constants[counter] = data[curLeftData]
                    counter += 1
                    break
                rowsInSegment = sum(data[curLeftData:curRightData])
                curRow += rowsInSegment
                curRightEdge += float(rowsInSegment) / data[curLeftData]
                self.rightIntervalEdges[counter] = curRightEdge
                self.cumulativeRows[counter] = curRow
                self.constants[counter] = data[curLeftData]
                curLeftData = curRightData 
                counter += 1
            else:
                self.rightIntervalEdges[counter] = curRightData
                self.constants[counter] = data[curLeftData]
                curRow += data[curLeftData] * (curRightData-curLeftData)
                self.cumulativeRows[counter] = curRow
                curLeftData = curRightData
                counter += 1
        self.rightIntervalEdges = self.rightIntervalEdges[:counter]
        self.cumulativeRows = self.cumulativeRows[:counter]
        self.constants = self.constants[:counter]
        
    cpdef __reduce__(self):
        return (arraysToPiecewiseConstantFunction, (list(self.rightIntervalEdges), list(self.cumulativeRows), list(self.constants)))
    
    cpdef double integrateFunction(self):
        curLeft = 0
        x = 0
        for i in range(len(self.constants)):
            curRight = self.rightIntervalEdges[i]
            x += self.constants[i]*(curRight-curLeft)
            curLeft = curRight
        return x
            
    cpdef double calculateInverse(self, double y):
        cdef long segment = min(bisect_left_double(self.cumulativeRows, y), len(self.constants)-1)
        rowsIntoCurSegment = y
        leftVal = 0
        if segment > 0:
            rowsIntoCurSegment -= self.cumulativeRows[segment - 1]
            leftVal = self.rightIntervalEdges[segment - 1]
        x = leftVal + rowsIntoCurSegment/float(self.constants[segment])
        return x
    
    cpdef double calculateValueAtPoint(self, double x):
        cdef long segment = min(bisect_left_double(self.rightIntervalEdges, x), len(self.constants)-1)
        return self.constants[segment]
    
    cpdef double calculateRowsAtPoint(self, double x):
        x = min(self.rightIntervalEdges[-1], x)
        cdef long segment = min(bisect_left_double(self.rightIntervalEdges, x), len(self.constants)-1)
        cdef double leftVal = 0
        cdef double cumulativeRows = 0
        if segment > 0:
            cumulativeRows += self.cumulativeRows[segment - 1]
            leftVal += self.rightIntervalEdges[segment - 1]
        cdef double constant = self.constants[segment]
        cumulativeRows += (x-leftVal)*constant
        return cumulativeRows
    
    cpdef PiecewiseConstantFunction prependFunc(self, PiecewiseConstantFunction other):
        newFunc = self.copy()
        numRowsToPrepend = other.getNumRows()
        numValuesToPrepend = other.rightIntervalEdges[-1]
        for i in range(len(newFunc.cumulativeRows)):
            newFunc.cumulativeRows[i] += numRowsToPrepend
            newFunc.rightIntervalEdges[i] += numValuesToPrepend
        newFunc.constants = np.concatenate((other.constants, newFunc.constants), axis=0)
        newFunc.rightIntervalEdges = np.concatenate((other.rightIntervalEdges, newFunc.rightIntervalEdges), axis=0)
        newFunc.cumulativeRows = np.concatenate((other.cumulativeRows, newFunc.cumulativeRows), axis=0)
        return newFunc
    
    cpdef PiecewiseConstantFunction leftTruncate(self, double x):
        if x >= self.rightIntervalEdges[-1]:
            return getEmptyFunction()
        cdef long startingSegment = min(bisect_right(self.rightIntervalEdges, x), len(self.constants)-1)
        startingSegmentLength = self.rightIntervalEdges[startingSegment] - x
        resultFunc = self.copy()
        resultFunc.constants = resultFunc.constants[startingSegment:]
        resultFunc.rightIntervalEdges = resultFunc.rightIntervalEdges[startingSegment:]
        resultFunc.cumulativeRows = resultFunc.cumulativeRows[startingSegment:]
        startingRightInterval = resultFunc.rightIntervalEdges[0] - startingSegmentLength
        startingRow = resultFunc.cumulativeRows[0] - startingSegmentLength*resultFunc.constants[0]
        for i in range(len(resultFunc.rightIntervalEdges)):
            resultFunc.rightIntervalEdges[i] -= startingRightInterval
            resultFunc.cumulativeRows[i] -= startingRow
        return resultFunc
    
    cpdef PiecewiseConstantFunction rightTruncateByRows(self, double y):
        endingSegment = min(bisect_left_double(self.cumulativeRows, y), len(self.constants)-1)
        resultFunc = self.copy()
        resultFunc.constants = resultFunc.constants[:endingSegment+1]
        resultFunc.cumulativeRows = resultFunc.cumulativeRows[:endingSegment+1]
        resultFunc.cumulativeRows[-1] = y
        resultFunc.rightIntervalEdges = resultFunc.rightIntervalEdges[:endingSegment+1]
        endingSegmentRows = y
        if endingSegment > 0:
            endingSegmentRows = y - resultFunc.cumulativeRows[-2]
        endingSegmentValues = endingSegmentRows/resultFunc.constants[-1]
        if endingSegment == 0:
            resultFunc.rightIntervalEdges[-1] = endingSegmentValues
        else:
            resultFunc.rightIntervalEdges[-1] = resultFunc.rightIntervalEdges[-2] + endingSegmentValues
            
        if len(resultFunc.constants) > 1:
            resultFunc.constants[-1] = min(resultFunc.constants[-1], resultFunc.cumulativeRows[-1] - resultFunc.cumulativeRows[-2])
        else:
            resultFunc.constants[-1] = min(resultFunc.constants[-1], resultFunc.cumulativeRows[-1])
        return resultFunc
        
    cpdef compressFunc(self, long numSegments = 0):
        if numSegments > 0:
            newConstants = []
            newRightIntervalEdges = []
            bins = np.array(calculateBinsExponential(ceil(self.rightIntervalEdges[-1]), numSegments), dtype=np.float64)
            bins[-1] = self.rightIntervalEdges[-1]
            newCumulativeRows = []
            maxRows = self.cumulativeRows[-1]
            left = 0
            cumulativeRows = 0
            for right in bins:
                if right == left:
                    continue
                constant = self.calculateValueAtPoint(left)
                newConstants.append(constant)
                if cumulativeRows + constant*(right-left) > maxRows:
                    right = left + (maxRows-cumulativeRows)/constant
                    newCumulativeRows.append(maxRows)
                    newRightIntervalEdges.append(right)
                    break
                cumulativeRows += constant*(right-left)
                newCumulativeRows.append(cumulativeRows)
                newRightIntervalEdges.append(right)
                left = right
            newFunc = getEmptyFunction()
            newFunc.constants = np.array(newConstants, dtype='double')
            newFunc.rightIntervalEdges = np.array(newRightIntervalEdges, dtype='double')
            newFunc.cumulativeRows = np.array(newCumulativeRows, dtype='double')
            self.constants = newFunc.constants
            self.rightIntervalEdges = newFunc.rightIntervalEdges
            self.cumulativeRows = newFunc.cumulativeRows
            
        duplicateItems = set()
        for i in range(len(self.constants)-1):
            if self.constants[i] == self.constants[i+1]:
                duplicateItems.add(i)
        
        newConstants = []
        newRightIntervalEdges = []
        newCumulativeRows = []
        for i in range(len(self.constants)):
            if i in duplicateItems:
                continue
            newConstants.append(self.constants[i])
            newRightIntervalEdges.append(self.rightIntervalEdges[i])
            newCumulativeRows.append(self.cumulativeRows[i])
        self.constants = np.array(newConstants, dtype='double')
        self.rightIntervalEdges = np.array(newRightIntervalEdges, dtype='double')
        self.cumulativeRows = np.array(newCumulativeRows, dtype='double')
        
    cpdef double getNumRows(self):
        return self.cumulativeRows[-1]
    
    cpdef double getSelfJoinSize(self):
        selfJoinSize = 0
        left = 0
        right = 0
        for i in range(len(self.constants)):
            right = self.rightIntervalEdges[i]
            selfJoinSize += (right-left)*(self.constants[i]**2)
            left = right
        return selfJoinSize
    
    cpdef setConstants(self, double[:] newConstants):
        self.constants = newConstants
        
    cpdef setRightIntervalEdges(self, double[:] newRightIntervalEdges):
        self.rightIntervalEdges = newRightIntervalEdges
        
    cpdef setCumulativeRows(self, double[:] newCumulativeRows):
        self.cumulativeRows = newCumulativeRows
    
    cpdef PiecewiseConstantFunction copy(self):
        newFunc = getEmptyFunction()
        newFunc.constants = np.array(self.constants)
        newFunc.rightIntervalEdges = np.array(self.rightIntervalEdges)
        newFunc.cumulativeRows = np.array(self.cumulativeRows)
        return newFunc
        
    cpdef printDiagnostics(self):
        curLeft = 0
        for i in range(len(self.constants)):
            curRight = self.rightIntervalEdges[i]
            print("Interval: [" + str(curLeft) + ", " + str(curRight) + ")    Constant: " + str(self.constants[i]) + "  Cumulative Rows: " + str(self.cumulativeRows[i]))
            curLeft = curRight
            
    cpdef double memory(self):
        footprint = len(self.rightIntervalEdges)*8 + len(self.constants)*8 + len(self.cumulativeRows)*8
        return footprint
