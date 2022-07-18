cdef class PiecewiseConstantFunction:
    cdef public double[:] constants
    cdef public double[:] rightIntervalEdges
    cdef public double[:] cumulativeRows
    
    cpdef double integrateFunction(self)
    cpdef double calculateInverse(self, double y)
    cpdef double calculateValueAtPoint(self, double x)
    cpdef double calculateRowsAtPoint(self, double x)
    cpdef double getNumRows(self)
    cpdef double getSelfJoinSize(self)
    cpdef setConstants(self, double[:] newConstants)
    cpdef setRightIntervalEdges(self, double[:] newRightIntervalEdges)
    cpdef setCumulativeRows(self, double[:] newCumulativeRows)
    cpdef PiecewiseConstantFunction prependFunc(self, PiecewiseConstantFunction other)
    cpdef PiecewiseConstantFunction leftTruncate(self, double x)
    cpdef PiecewiseConstantFunction rightTruncateByRows(self, double y)
    cpdef compressFunc(self, long numSegments = *)
    cpdef __reduce__(self)
    cpdef PiecewiseConstantFunction copy(self)
    cpdef printDiagnostics(self)
    cpdef double memory(self)
    
cpdef PiecewiseConstantFunction getEmptyFunction()
    
cpdef double pairwiseFunctionDifference(PiecewiseConstantFunction function1, PiecewiseConstantFunction function2)

cpdef double estimatedFunctionWeight(PiecewiseConstantFunction function)

cpdef PiecewiseConstantFunction pointwiseFunctionMax(PiecewiseConstantFunction[:] functions)

cpdef PiecewiseConstantFunction pointwisePDFMax(PiecewiseConstantFunction[:] functions)

cpdef PiecewiseConstantFunction pointwiseFunctionMin(PiecewiseConstantFunction[:] functions)

cpdef PiecewiseConstantFunction pointwiseFunctionMult(PiecewiseConstantFunction[:] functions)

cpdef long[:] calculateBinsRelativeError(long[:] data, double relativeError)

cpdef long[:] calculateBinsLinear(long numValues, long numSegments)

cpdef long[:] calculateBinsExponential(long numValues, long numSegments)
