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

```
SafeBound(tables, tableNames, tableJoinCols, relativeErrorPerSegment, originalFilterCols = [], 
                     numBuckets = 25, numEqualityOutliers=500, FKtoKDict = dict(),
                     numOutliers = 5, trackNulls=True, trackTriGrams=True, numCores=12, groupingMethod = "CompleteClustering",
                     modelCDF=True, verbose=False)
```

They split roughly into three categories; 1) required data inputs 2) tuning knobs 3) experimental configurations. 

The former are ***tables, tableNames, tableJoinCols, originalFilterCols and FKtoKDict*** which define the data and schema. All columns which will be used in an equi-join must be declared in the ***tableJoinCols*** parameter, and all columns which will be filtered on (e.g. 'R.A=1') must be declared in ***originalFilterCols***. The final parameter ***FKtoKDict*** is used to define FK-PK relationships and does not need to cover all joins in the target workload. It simply allows for statistics collection across FK-PK joins. Note: the graph defined on the tables by the FK-K dictionary must be acyclic.

The primary tuning knobs are ***relativeErrroPerSegment, numBuckets, numEqualityOutliers, and NumOutliers***. These control CDF modeling accuracy, histogram granularity, MCV list length, and the aggressiveness of the clustering operation, respectively. ***NumCores*** tunes the number of processes and amount of memroy used when building the statistics, but does not affect the statistics once constructed. ***trackNulls and trackTriGrams*** simply remove the statistics for NULL/NOT NULL and LIKE predicates for workloads where these do not apply.

Lastly, ***groupingMethod and ModelCDF*** are experimental knobs which should be left to their defaults in practice.

When querying, the parameters are much simpler.

```
SafeBound.functionalFrequencyBound(self, joinQueryGraph, dbConn = None, verbose = 0)
```
The ***JoinQueryGraph*** defines the target query's structure and predicates and is, of course, required. Further, the definition of the JoinQueryGraph must line up with the tables and columns defined during the statistics construction. ***dbConn*** is an experimental parameter which should be ignored. The result of this function will be a nonnegative double which upper bounds the cardinality of the query.


**JoinQueryGraph**

As stated above, the [JoinQueryGraph](https://github.com/AnonymousSigmod2023/SafeBound/blob/main/Source/JoinGraphUtils.pyx) class defines a particular SQL query. The constructor requires no inputs, then aliases/joins/predicates are incrementally added to the query. Here is an example usage which finds all actors which have acted in both 1970 and 2010:

```
query = JoinQueryGraph(
query.addAlias("ActorName", "an")
query.addAlias("ActedIn", "ai1")
query.addAlias("ActedIn", "ai2")
query.addJoin("an", "Id", "ai1", "ActorId")
query.addJoin("an", "Id", "ai2", "ActorId")
query.addPredicate("ai1", "Year", "=", "1970")
query.addPredicate("ai2", "Year", "=", "2010")
query.buildJoinGraph()
query.printJoinGraph()
```

There is experimental parsing code which can be found in [SQLParser.py](https://github.com/AnonymousSigmod2023/SafeBound/blob/main/Source/SQLParser.py) that transforms a file with many SQL queries to a list of JoinQueryGraphs. However, it is currently very limited and requires SQL queries which are nicely structured.

# Current Limitations

SafeBound can currently handle acyclic queries with single-attribute inner joins and equality, range, LIKE, or IN predicates. Further, it can technically handle cyclic queries, although the estimates are somewhat limited. It does not support: outer joins, negation, theta joins, or multi-attribute joins. 


# Reproducibility
This section assumes a linux (specifically Ubuntu) environment. Other linux distributions should be easy to adjust the instructions for, but Windows and MacOS are not supported. 

***Building SafeBound Library***
1) Set up the conda environment in order to build SafeBound using the environment.yml file.

```
conda env create
conda activate TestEnv
```

2) Build the pybloomfilter package.

```
sudo apt-get update
sudo apt-get install gcc
cd pybloomfiltermmap3
python setup.py install
cd ..
``` 

3) Build the SafeBound package

```
sudo apt-get install g++
cd Source
python CythonBuild.py build_ext --inplace
cd ..
```

At this point, the SafeBound library should be ready for use. An example usage can be found in the "ExampleUsage" notebook. 

***Setting Up Benchmarks***

The following steps are specific to recreating the experimental environment from the paper. Specifically, setting up the Postgres instance which will be used to obtain runtime results.

4) Setup postgres cluster. If one already exists on the system, then this step can likely be skipped.

```
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install postgresql-13
sudo apt-get -y install postgresql-server-dev-13
sudo pg_ctlcluster 13 main start
```
Once it is installed, modify the postgresql.conf file to set reasonable configuration parameters. Specifically, we set the shared memory to
4GB, worker memory to 2GB, OS cache size to 32 GB, and max parallel workers to 6.


To make the remainder of the commands run smoothly, you have to create a user with the same username as your linux username. This allows for the "psql" command to be run without sudo.

```sudo -u postgres createuser [USERNAME] –interactive```

5) Install the pg_hint_plan extension for postgres. This is used for injecting cardinality estimates into postgres' query optimizer.

```
git clone https://github.com/ossc-db/pg_hint_plan.git
cd pg_hint_plan
git checkout PG13
sudo make
sudo make install
```

6) Create the JOB benchmark databases. This command may run for several minutes as it creates a version of the IMDB database for each benchmark JOBLight, JOBLightRanges, and JOBM.

```
cd ..
cd ..
bash CreateJOBBenchmark.bash
```

7) Create the Stats benchmark database.

```
bash CreateStatsBenchmark.bash
```

***Running Experiments***

The experiments are run from three main files: ``BuildExperiments.py``, ``InferenceExperiments.py``, and ``RuntimeExperiments.py``. Note that these will take significant time to complete. Roughly, BuildExperiments should take 4-8 hours. InferenceExperiments should take 10-30 minutes. RuntimeExperiments should take 1-2 days.

Restricting the experiments to the JOBLight and Stats benchmarks should significantly lower this time, and this can be done by modifying the ``benchmarks`` parameter in each of these files. Further, reducing the ``numberOfRuns`` parameter in RuntimeExperiments.py will of course reduce the overall time.

In order to run them, simply do the following commands in order:

8) ``python Experiments/BuildExperiments.py``
9) ``python Experiments/InferenceExperiments.py``
10) ``python Experiments/RuntimeExperiments.py``

After these are run, results should be available in the Data/Results directory. The main visualizations can then be generated with the jupyter notebooks BuildGraph, InferenceGraphs, and RuntimeGraphs.



# References
This project relies on several others for crucial aspects. Some of these have been directly included in the repository for the sake of stability and reproducibility. For the bloom filters which represent our MCV lists, we use the bloom filters provided by the [pybloomfiltermmap3](https://github.com/prashnts/pybloomfiltermmap3) project. In order to integrate estimates into Postgresql, we take advantage of the pg_hint_plan plug-in which can be found [here](https://github.com/ossc-db/pg_hint_plan). We include comparison with [Postgres](https://www.postgresql.org/)'s internal optimizer.

We compare against two benchmarks. The first, the Join Order Benchmark, was originally proposed by [Viktor Leis et al.](https://github.com/gregrahn/join-order-benchmark), but we use the query workloads from later [work](https://github.com/neurocard/neurocard/tree/master/neurocard/queries). The second is the STATS benchmark which can be found [here](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark).

# Contact

If you're interested in this work, have questions, or run into issues with the code base, feel free to reach out to me at kylebd99@gmail.com ! 


