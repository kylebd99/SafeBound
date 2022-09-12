# Building The Source
SafeBound is built on Cython which is a semi-compiled version of Python. This affects the low-level PiecewiseConstantFunction class the most.

However, more generally, we note that, due to Cython, SafeBound requires compilation unlike other Python libraries. This is done using the following command:

```
python CythonBuild.py build_ext --inplace
```

# File Structure
The primary files in this directory are SafeBoundUtils.pyx, HistogramUtils.pyx, PiecewiseConstantFunctionUtils.pyx, PiecewiseConstantFunctionUtils.pyd, and JoinGraphUtils.pyx.

PiecewiseConstantFunctionUtils.py(xd) handles the lowest-level operations on piece-wise functions. HistogramUtils.pyx manages the statistics collected for each filter column and produces cumulative degree functions given predicates. Lastly, SafeBoundUtils.pyx handles the high-level statistics collection and the bound calculation after predicates are applied.


