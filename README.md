# SafeBound
<p align="center">
<img src="https://user-images.githubusercontent.com/108958308/189722447-40a55315-92a1-45dd-9793-ba2abe3bfdae.png"  width="500" height="300">
</p>
This repository contains the experimental code for SafeBound, a practical cardinality bounding system. As opposed to traditional cardinality estimators, SafeBound produces an upper bound on the number of rows which a query can generate. The goal is for query optimizers which use these upper bounds to produce more robust, less optimistic query plans. 

SafeBound is based off of theoretical work on the Degree Sequence Bound which can be found [here](https://arxiv.org/pdf/2201.04166). The theoretical details and broader design of SafeBound will be found in a forthcoming paper.


# Usage

There are two external-facing classes in this repository, `SafeBound` and `JoinQueryGraph`. The former constructs all statistics on the data and produces cardinality bounds while the latter defines a target query.

**SafeBound**

The constructor for [SafeBound](https://github.com/AnonymousSigmod2023/SafeBound/blob/main/Source/SafeBoundUtils.pyx) has many parameters:

``
SafeBound(tableDFs, tableNames, tableJoinCols, relativeErrorPerSegment, originalFilterCols = [], 
                     numBuckets = 25, numEqualityOutliers=500, FKtoKDict = dict(),
                     numOutliers = 5, trackNulls=True, trackBiGrams=True, numCores=12, groupingMethod = "CompleteClustering",
                     modelCDF=True, verbose=False)
``

They split roughly into three categories; 1) required data inputs 2) tuning knobs 3) experimental configurations. 

The former are ***tableDFs, tableNames, tableJoinCols, originalFilterCols and FKtoKDict*** which define the data and schema. All columns which will be used in an equi-join must be declared in the ***tableJoinCols*** parameter, and all columns which will be filtered on (e.g. 'R.A=1') must be declared in ***originalFilterCols***. The final parameter ***FKtoKDict*** is used to define FK-PK relationships and does not need to cover all joins in the target workload. It simply allows for statistics collection across FK-PK joins. Note: the graph defined on the tables by the FK-K dictionary must be acyclic.

The primary tuning knobs are ***relativeErrroPerSegment, numBuckets, numEqualityOutliers, and NumOutliers***. These control CDF modeling accuracy, histogram granularity, MCV list length, and the aggressiveness of the clustering operation, respectively. ***NumCores*** tunes the number of processes used when building the statistics, but does not affect the statistics once constructed. ***trackNulls and trackBiGrams*** simply remove the statistics for NULL/NOT NULL and LIKE predicates for workloads where these do not apply.

Lastly, ***groupingMethod and ModelCDF*** are experimental knobs which should be left to their defaults in practice.

When querying, the parameters are much simpler.

``
SafeBound.functionalFrequencyBound(self, joinQueryGraph, dbConn = None, verbose = 0)
``
The ***JoinQueryGraph*** defines the target query's structure and predicates and is, of course, required. Further, the definition of the JoinQueryGraph must line up with the tables and columns defined during the statistics construction. ***dbConn*** is an experimental parameter which should be ignored. The result of this function will be a nonnegative double which upper bounds the cardinality of the query.


**JoinQueryGraph**

As stated above, the [JoinQueryGraph](https://github.com/AnonymousSigmod2023/SafeBound/blob/main/Source/JoinGraphUtils.pyx) class defines a particular SQL query. The constructor requires no inputs, then aliases/joins/predicates are incrementally added to the query. Here is an example usage which finds all actors which have acted in both 1970 and 2010:

``
query = JoinQueryGraph()  \n
query.addAlias("ActorName", "an")  
query.addAlias("ActedIn", "ai1")  
query.addAlias("ActedIn", "ai2")  
query.addJoin("an", "Id", "ai1", "ActorId")  
query.addJoin("an", "Id", "ai2", "ActorId")  
query.addPredicate("ai1", "Year", "=", "1970")  
query.addPredicate("ai2", "Year", "=", "2010")  
query.buildJoinGraph()  
query.printJoinGraph()
``

There is experimental parsing code which can be found in [SQLParser.py](https://github.com/AnonymousSigmod2023/SafeBound/blob/main/Source/SQLParser.py) that transforms a file with many SQL queries to a list of JoinQueryGraphs. However, it is currently very limited and requires SQL queries which are nicely structured.

















