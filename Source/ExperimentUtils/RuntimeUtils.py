import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import re
import sys
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +'/'
sys.path.append(rootFileDirectory + 'Source')
from DBConnectionUtils import *
from SafeBoundUtils import *
from SimplicityImplementation import *
from SQLParser import *

def reset_cache():
    os.system("sudo sh -c '/usr/bin/echo 3 > /proc/sys/vm/drop_caches'")
    os.system("sudo systemctl restart postgresql-13")

def evaluate_runtime_safe_bound(statsFile, 
                                benchmark,
                                queryJGs,
                                dbConn,
                                outputFile,
                                performTableScan, runs):
    dbConn.changeStatisticsTarget(100)
    stats = pickle.load(open(statsFile, 'rb'))
    queryLabels = []
    runLabels = []
    inferenceTimes = []
    runtimes = []
    for j in range(runs):
        reset_cache()
        dbConn.reset()
        for i, queryJG in enumerate(queryJGs):
            print(str(j) + "," + str(i))
            queryHints = []
            inferenceStart = datetime.now()
            for subQuery in queryJG.getIntermediateQueries():
                if performTableScan:
                    estimate = stats.functionalFrequencyBound(subQuery, dbConn = dbConn)
                else:
                    estimate = stats.functionalFrequencyBound(subQuery)
                tables = list(subQuery.vertexDict.keys())
                hint = JoinHint(tables, estimate)
                queryHints.append(hint)
            inferenceEnd = datetime.now()
            dbConn.printQueryPlan(queryJG, queryHints)

            if i % 10 == 1:
                print("SafeBound Runtime Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
            queryLabels.append(i)
            runLabels.append(j)
            inferenceTimes.append((inferenceEnd-inferenceStart).total_seconds())
            runtimeStart = datetime.now()
            actual = 0
            estimate, actual = dbConn.getSizeEstimateAndActual(queryJG, queryHints)
            runtimeEnd = datetime.now()
            runtimes.append((runtimeEnd-runtimeStart).total_seconds())
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['RunLabel'] = runLabels
    inferenceResults['InferenceTime'] = inferenceTimes
    inferenceResults['Runtime'] = runtimes
    inferenceResults['StatsSize'] = stats.memory()
    inferenceResults.to_csv(outputFile)
    
    
def evaluate_runtime_postgres(benchmark,
                              queryJGs,
                              dbConn,
                              statisticsTarget,
                              outputFile, runs):
    dbConn.changeStatisticsTarget(statisticsTarget)
    queryLabels = []
    runLabels = []
    inferenceTimes = []
    runtimes = []
    for j in range(runs):
        reset_cache()
        dbConn.reset()
        for i, queryJG in enumerate(queryJGs):
            print(str(j) + "," + str(i))
            queryHints = []
            for subQuery in queryJG.getIntermediateQueries():
                estimate = dbConn.getSizeEstimate(subQuery)
                tables = list(subQuery.vertexDict.keys())
                hint = JoinHint(tables, estimate)
                queryHints.append(hint)

            inferenceStart = datetime.now()
            dbConn.printQueryPlan(queryJG, queryHints)
            inferenceEnd = datetime.now()
            if i % 10 == 1:
                print("Postgres Runtime Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
            queryLabels.append(i)
            runLabels.append(j)
            inferenceTimes.append((inferenceEnd-inferenceStart).total_seconds())
            runtimeStart = datetime.now()
            estimate, actual = dbConn.getSizeEstimateAndActual(queryJG, queryHints)
            runtimeEnd = datetime.now()
            runtimes.append((runtimeEnd-runtimeStart).total_seconds())
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['RunLabel'] = runLabels
    inferenceResults['InferenceTime'] = inferenceTimes
    inferenceResults['Runtime'] = runtimes
    inferenceResults['StatsSize'] = dbConn.memory()
    inferenceResults.to_csv(outputFile)
    
    
def removeFKJoins(queryJG):
    userIDFKNames = ["USERID", "OWNERUSERID"]
    postIDFKNames = ["POSTID", "RELATEDPOSTID"]
    
    newJG = JoinQueryGraph()
    
    for vertex in queryJG.vertexDict.values():
        newJG.addAlias(vertex.tableName, vertex.alias)
        
    for vertex in queryJG.vertexDict.values():
        for pred in vertex.predicates:
            newJG.addPredicate(vertex.alias, pred.colName, pred.predType, pred.compValue)
        for i in range(len(vertex.outputJoinCols)):
            if vertex.outputJoinCols[i] == "MOVIE_ID" and vertex.edgeJoinCols[i] == "MOVIE_ID":  
                newJG.addAlias("TITLE", "t")  
                newJG.addJoin('t', 'id', vertex.alias, 'MOVIE_ID')
                newJG.addJoin('t', 'id', vertex.edgeAliases[i], 'MOVIE_ID')
            elif vertex.outputJoinCols[i] in userIDFKNames and vertex.edgeJoinCols[i] in userIDFKNames:  
                if "u" not in queryJG.tableDict:
                    newJG.addAlias("USERS", "u")
                newJG.addJoin('u', 'id', vertex.alias, vertex.outputJoinCols[i])
                newJG.addJoin('u', 'id', vertex.edgeAliases[i], vertex.edgeJoinCols[i])
            elif vertex.outputJoinCols[i] in postIDFKNames and vertex.edgeJoinCols[i] in postIDFKNames:  
                if "p" not in queryJG.tableDict:
                    newJG.addAlias("POSTS", "p")
                newJG.addJoin('p', 'id', vertex.alias, vertex.outputJoinCols[i])
                newJG.addJoin('p', 'id', vertex.edgeAliases[i], vertex.edgeJoinCols[i])
            else: 
                newJG.addJoin(vertex.alias, vertex.outputJoinCols[i], vertex.edgeAliases[i], vertex.edgeJoinCols[i])
    newJG.buildJoinGraph(addImpliedJoins=False)
    return newJG.getSQLQuery(countStar=True)
    
    
def fixStatsCases(query):
    namesToFix = [ "VoteTypeId",  "CreationDate", "BountyAmount", "PostHistoryTypeId",
                   'Score', 'ViewCount', 'OwnerUserId', 'AnswerCount', 'CommentCount', 'FavoriteCount', 'LastEditorUserId',
                      'Reputation', 'Views', 'UpVotes', 'DownVotes', 'RelatedPostId', 'LinkTypeId', 'PostTypeId',
                      'Count', 'ExcerptPostId', "UserId", "PostId","Id", "postHistory","Date", "postLinks"]
    for name in namesToFix:
        query = query.replace(name.lower(), name)
    query = query.replace("Count(*)", "COUNT(*)")
    
    query = re.sub("('....-..-.. ..:..:..')", "\\1::timestamp", query) 
    return query

def evaluate_runtime_bayes_card(ensembleDirectory, 
                                  benchmark,
                                  queryJGs,
                                  dbConn,
                                  outputFile, runs):
    dbConn.changeStatisticsTarget(100)
    schema = None
    if benchmark == 'JOBLight':
        csvPath = rootDirectory + "Data/JOB"
        schema = gen_job_light_imdb_schema(csvPath)
    elif benchmark == 'JOBLightRanges':
        return
    elif benchmark == 'JOBM':
        return
    elif benchmark == 'Stats':
        csvPath = rootDirectory + "Data/Stats/stats_simplified_bayes_card"
        schema = gen_stats_light_schema(csvPath)
    
    modelSize = 0
    bn_ensemble = BN_ensemble(schema)
    for file in os.listdir(ensembleDirectory):
        if file.endswith(".pkl"):
            with open(ensembleDirectory + file, "rb") as f:
                bn = pickle.load(f)
                bn.infer_algo = "exact-jit"
                bn.init_inference_method()
            bn_ensemble.bns.append(bn)
            modelSize += os.path.getsize(ensembleDirectory + file)
    
    queryLabels = []
    runLabels = []
    inferenceTimes = []
    runtimes = []
    for j in range(runs):
        reset_cache()
        dbConn.reset()
        for i, queryJG in enumerate(queryJGs):
            print(str(j) + "," + str(i))
            if i % 10 == 1:
                print("BayesCard Runtime Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
            queryLabels.append(i)
            runLabels.append(j)
            subQueryJGs = queryJG.getIntermediateQueries()
            subQuerySQLs = [removeFKJoins(query) for query in subQueryJGs]
            tempSQLFilePath = rootDirectory +'Workloads/SubQueries.sql'
            tempSQLFile = open(tempSQLFilePath, 'w')
            for query in subQuerySQLs:
                if benchmark == 'Stats':
                    tempSQLFile.write(fixStatsCases(query.lower()) + '\n')
                else:
                    tempSQLFile.write(query.lower() + '\n')
            tempSQLFile.close()

            parsedSubQueries, _ = prepare_join_queries(schema, ensembleDirectory, pairwise_rdc_path=None, 
                                                        query_filename=tempSQLFilePath, true_card_exist=False)
            subQueries = bn_ensemble.parse_query_all(parsedSubQueries)

            queryHints = []
            inferenceStart = datetime.now()
            for subQueryIndex, subQuery in enumerate(subQueries):
                tables = list(subQueryJGs[subQueryIndex].vertexDict.keys())
                estimate = bn_ensemble.cardinality(subQuery)
                hint = JoinHint(tables, estimate)
                queryHints.append(hint)
            inferenceEnd = datetime.now()

            dbConn.printQueryPlan(queryJG, queryHints)
            inferenceTimes.append((inferenceEnd-inferenceStart).total_seconds())
            runtimeStart = datetime.now()
            actual = 0
            estimate, actual = dbConn.getSizeEstimateAndActual(queryJG, queryHints)
            runtimeEnd = datetime.now()
            runtimes.append((runtimeEnd-runtimeStart).total_seconds())
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['RunLabel'] = runLabels
    inferenceResults['InferenceTime'] = inferenceTimes
    inferenceResults['Runtime'] = runtimes
    inferenceResults['StatsSize'] = modelSize
    inferenceResults.to_csv(outputFile)
    
    
def evaluate_runtime_true_cardinality(statsFile, benchmark, queryJGs,
                                        dbConn, outputFile, runs):
    dbConn.changeStatisticsTarget(100)
    queryLabels = []
    runLabels = []
    inferenceTimes = []
    runtimes = []
    allQueryHints = pickle.load(open(statsFile, "rb"))
    for j in range(runs):
        reset_cache()
        dbConn.reset()
        for i, queryJG in enumerate(queryJGs):
            print(str(j) + "," + str(i))
            queryHints = allQueryHints[i]
            if i % 10 == 1:
                print("True Card Runtime Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
            queryLabels.append(i)
            runLabels.append(j)
            dbConn.printQueryPlan(queryJG, queryHints)
            runtimeStart = datetime.now()
            estimate, actual = dbConn.getSizeEstimateAndActual(queryJG, queryHints)
            runtimeEnd = datetime.now()
            runtimes.append((runtimeEnd-runtimeStart).total_seconds())
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['RunLabel'] = runLabels
    inferenceResults['InferenceTime'] = -1
    inferenceResults['Runtime'] = runtimes
    inferenceResults['StatsSize'] = -1
    inferenceResults.to_csv(outputFile)
    
    
def evaluate_runtime_pessimistic_cardinality_estimation(statsFile, benchmark, queryJGs,
                                                        dbConn, outputFile, runs):
    dbConn.changeStatisticsTarget(100)
    queryLabels = []
    runLabels = []
    inferenceTimes = []
    runtimes = []
    for j in range(runs):
        reset_cache()
        dbConn.reset()
        for i, queryJG in enumerate(queryJGs):
            print(str(j) + "," + str(i))
            if i % 10 == 1:
                print("PessEst Runtime Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
            queryLabels.append(i)
            runLabels.append(j)
            queryHints = []
            hintFile = '/home/ec2-user/FrequencyBounds/pqo-opensource/output/results/' + benchmark + "_" + str(i) + "_SubQueryBounds.txt"

            for hintString in open(hintFile, "r").readlines():
                relations, estimate = hintString.split("|")
                relations = relations.split(",")
                estimate = int(estimate)
                hint = JoinHint(relations, estimate)
                queryHints.append(hint)
            inferenceTimeFile = '/home/ec2-user/FrequencyBounds/pqo-opensource/output/results/' + benchmark + "_" + str(i) + "_sketch_preprocessing.txt"
            inferenceTime = float(open(inferenceTimeFile, "r").readline())
            inferenceTimes.append(inferenceTime)
            dbConn.printQueryPlan(queryJG, queryHints)
            runtimeStart = datetime.now()
            estimate, actual = dbConn.getSizeEstimateAndActual(queryJG, queryHints)
            runtimeEnd = datetime.now()
            runtimes.append((runtimeEnd-runtimeStart).total_seconds())
    
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['RunLabel'] = runLabels
    inferenceResults['InferenceTime'] = inferenceTimes 
    inferenceResults['Runtime'] = runtimes
    inferenceResults['StatsSize'] = -1
    inferenceResults.to_csv(outputFile)
    
    
def evaluate_runtime_neurocard(benchmark, queryJGs, dbConn, outputFile, runs):
    if benchmark == "Stats":
        return
    
    hintDF = pd.read_csv("/home/ec2-user/FrequencyBounds/Workloads/NeuroCardHintIndex_" + benchmark + ".csv", delimiter='|', names=["QueryLabel", "Tables"])
    estimatesDF = pd.read_csv("/home/ec2-user/FrequencyBounds/Data/Results/NeuroCard_Inference_" + benchmark + "SubQueries.csv")
    hintDF["Tables"] = hintDF["Tables"].apply(lambda x: x.split(","))
    hintDF = hintDF.join(estimatesDF)
    allQueryInferenceTimes = [0 for _ in range(max(hintDF["QueryLabel"])+1)]
    allQueryHints = [[] for _ in range(max(hintDF["QueryLabel"])+1)]
    for i in range(len(hintDF)):
        queryLabel = hintDF["QueryLabel"].iloc[i]
        allQueryHints[queryLabel].append(JoinHint(hintDF["Tables"].iloc[i], hintDF["Estimate"].iloc[i]))
        allQueryInferenceTimes[queryLabel] += hintDF["InferenceTime"].iloc[i]
    
    inferenceTimes = []
    dbConn.changeStatisticsTarget(100)
    queryLabels = []
    runLabels = []
    runtimes = []
    for j in range(runs):
        reset_cache()
        dbConn.reset()
        for i, queryJG in enumerate(queryJGs):
            print(str(j) + "," + str(i))
            inferenceTimes.append(allQueryInferenceTimes[i])
            queryHints = allQueryHints[i]
            if i % 10 == 1:
                print("NeuroCard Runtime Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
            queryLabels.append(i)
            runLabels.append(j)
            dbConn.printQueryPlan(queryJG, queryHints)
            runtimeStart = datetime.now()
            estimate, actual = dbConn.getSizeEstimateAndActual(queryJG, queryHints)
            runtimeEnd = datetime.now()
            runtimes.append((runtimeEnd-runtimeStart).total_seconds())
    
    statsSize = 0
    if benchmark == "JOBM":
        statsSize = 27.3 * 10**6
    elif benchmark == "JOBLight":
        statsSize = 3.8 * 10**6
    elif benchmark == "JOBLightRanges":
        statsSize = 4.1 * 10**6
    
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['RunLabel'] = runLabels
    inferenceResults['InferenceTime'] = inferenceTimes
    inferenceResults['Runtime'] = runtimes
    inferenceResults['StatsSize'] = statsSize
    inferenceResults.to_csv(outputFile)
    
def evaluate_runtime(method = 'SafeBound', 
                    statsFile = '../../StatObjects/SafeBound_JOBLight_1.csv',
                    benchmark = 'JOBLight',
                    outputFile = '../../Data/Results/SafeBound_Inference_JOBLight_1.csv',
                    statisticsTarget = None,
                    runs = 5):
    if method == 'Postgres2D':
        benchmark += "2D"
    
    queryJGs = None        
    dbConn = None
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
        dbConn = getDBConn('stats')
    elif benchmark == 'JOBLight2D':
        queryJGs =  SQLFileToJoinQueryGraphs(rootFileDirectory + 'Workloads/JOBLightQueries.sql')
        dbConn = getDBConn('imdblight2d')
    elif benchmark == 'JOBLightRanges2D':
        queryJGs =  SQLFileToJoinQueryGraphs(rootFileDirectory + 'Workloads/JOBLightRangesQueries.sql')
        dbConn = getDBConn('imdblightranges2d')
    elif benchmark == 'JOBM2D':
        queryJGs =  SQLFileToJoinQueryGraphs(rootFileDirectory + 'Workloads/JOBMQueries.sql')
        dbConn = getDBConn('imdbm')
    elif benchmark == 'Stats2D':
        queryJGs =  SQLFileToJoinQueryGraphs(rootFileDirectory + 'Workloads/StatsQueries.sql')
        dbConn = getDBConn('stats2d')
    elif benchmark == 'StatsPK':
        queryJGs =  SQLFileToJoinQueryGraphs(rootFileDirectory + 'Workloads/StatsQueries.sql')
        dbConn = getDBConn('statspk')
    
    if method == 'SafeBound':
        evaluate_runtime_safe_bound(statsFile, benchmark, queryJGs, dbConn, outputFile, performTableScan = False, runs=runs)
    elif method == "SafeBoundScan":
        evaluate_runtime_safe_bound(statsFile, benchmark, queryJGs, dbConn, outputFile, performTableScan = True, runs=runs)
    elif method == 'Postgres':
        evaluate_runtime_postgres(benchmark, queryJGs, dbConn, statisticsTarget, outputFile, runs=runs)
    elif method == 'Postgres2D':
        evaluate_runtime_postgres(benchmark, queryJGs, dbConn, statisticsTarget, outputFile, runs=runs)
    elif method == 'BayesCard':
        evaluate_runtime_bayes_card(statsFile, benchmark, queryJGs,  dbConn, outputFile, runs=runs)
    elif method == 'TrueCardinality':
        evaluate_runtime_true_cardinality(statsFile, benchmark, queryJGs, dbConn, outputFile, runs=runs)
    elif method == 'PessemisticCardinality':
        evaluate_runtime_pessimistic_cardinality_estimation(statsFile, benchmark, queryJGs, dbConn, outputFile, runs=runs)
    elif method == 'NeuroCard':
        evaluate_runtime_neurocard(benchmark, queryJGs, dbConn, outputFile, runs=runs)
