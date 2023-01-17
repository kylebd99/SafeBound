import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import sys
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +'/'
sys.path.append(rootFileDirectory + 'Source')
from DBConnectionUtils import *
from SafeBoundUtils import *
from SQLParser import *
from SimplicityImplementation import *
sys.path.append(rootFileDirectory + 'bayescard')
from Schemas.stats.schema import gen_stats_light_schema
from Schemas.imdb.schema import gen_job_light_imdb_schema    
from DataPrepare.query_prepare_BayesCard import prepare_join_queries
from Models.BN_ensemble_model import BN_ensemble


def evaluate_inference_safe_bound(statsFile, 
                                  benchmark,
                                  outputFile):
    stats = pickle.load(open(statsFile, 'rb'))
    
    queryJGs = None    
    if benchmark == 'JOBLight':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightQueries.sql')
    elif benchmark == 'JOBLightRanges':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightRangesQueries.sql')
    elif benchmark == 'JOBM':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBMQueries.sql')
    elif benchmark == 'Stats':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/StatsQueries.sql')

    queryLabels = []
    inferenceTimes = []
    cardinalityBounds = []
    for i, queryJG in enumerate(queryJGs):
        if i % 10 == 1:
            print("SafeBound Inference Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
        queryLabels.append(i)
        inferenceStart = datetime.now()
        bound = stats.functionalFrequencyBound(queryJG)
        inferenceEnd = datetime.now()
        inferenceTimes.append((inferenceEnd-inferenceStart).total_seconds())
        cardinalityBounds.append(bound)
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['InferenceTime'] = inferenceTimes
    inferenceResults['Estimate'] = cardinalityBounds
    inferenceResults['StatsSize'] = stats.memory()
    inferenceResults.to_csv(outputFile)
    
    
    
    
def evaluate_inference_simplicity(statsFile, 
                                  benchmark,
                                  outputFile):
    stats = pickle.load(open(statsFile, 'rb'))
    queryJGs = None    
    dbConn = None
    if benchmark == 'JOBLight':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightQueries.sql')
        dbConn = getDBConn('imdblight')
    elif benchmark == 'JOBLightRanges':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightRangesQueries.sql')
        dbConn = getDBConn('imdblightranges')
    elif benchmark == 'JOBM':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBMQueries.sql')
        dbConn = getDBConn('imdbm')
    elif benchmark == 'Stats':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/StatsQueries.sql')
        dbConn = getDBConn('stats')
    
    queryLabels = []
    inferenceTimes = []
    cardinalityBounds = []
    for i, queryJG in enumerate(queryJGs):
        if i % 10 == 1:
            print("Simplicity Inference Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
        queryLabels.append(i)
        inferenceStart = datetime.now()
        bound = stats.getSimplicityBound(queryJG, dbConn)
        inferenceEnd = datetime.now()
        inferenceTimes.append((inferenceEnd-inferenceStart).total_seconds())
        cardinalityBounds.append(bound)
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['InferenceTime'] = inferenceTimes
    inferenceResults['Estimate'] = cardinalityBounds
    inferenceResults['StatsSize'] = stats.memory()
    inferenceResults.to_csv(outputFile)
    
def getRelevantPreds(joinQueryGraph, vertex, FKtoKDict):
    relevantFilterColsDict = dict()
    reachableAliases = set([vertex.alias])
    if vertex.tableName in FKtoKDict:
        FKEdges = FKtoKDict[vertex.tableName]
        for joinCol in vertex.outputJoinCols:
            FKEdgesForJoinCol = [x for x in FKEdges if x[0] == joinCol]
            equalityClassMembers = joinQueryGraph.equalityClasses[joinQueryGraph.equalityDict[vertex.alias + "." + vertex.tableName+"."+joinCol]]
            for edge in FKEdgesForJoinCol:
                for member in equalityClassMembers:
                    parts = member.split('.')
                    if edge[1] ==  parts[2] and edge[2] == joinQueryGraph.tableDict[parts[0]]:
                        reachableAliases.add(parts[0])
    relevantPreds = []
    for alias in reachableAliases:
        for pred in joinQueryGraph.vertexDict[alias].predicates:
            suffix = ""
            prefix = ""
            if alias != vertex.alias:
                suffix = "_fk_pk"
                prefix = joinQueryGraph.tableDict[alias] + "_"
            relevantPreds.append(Predicate(prefix + pred.colName + suffix, pred.predType, pred.compValue))
        
        if alias != vertex.alias:
            joinQueryGraph.vertexDict[alias].predicates = []
    return relevantPreds


def propagatePKPredicates(originalQuery, FKtoKDict):
    query = originalQuery.copy()
    for vertex in query.vertexDict.values():
        vertex.predicates = getRelevantPreds(query, vertex, FKtoKDict)
    return query

    
def removePKJoins(query, FKtoKDict):
    query = propagatePKPredicates(query, FKtoKDict)
    return query

def evaluate_inference_postgres(benchmark, 
                                outputFile,
                                statisticsTarget):
    
    IMDB_FKtoKDict = {"aka_title":[["movie_id", "id", "title"], ["kind_id", "id", "kind_type"]],
         "cast_info":[["movie_id", "id", "title"],
                      ["person_role_id", "id", "char_name"],
                      ["role_id", "id", "role_type"]],
        "movie_companies":[["movie_id", "id", "title"],
                          ["company_id", "id", "company_name"],
                          ["company_type_id", "id", "company_type"]],
        "movie_info":[["movie_id", "id", "title"],
                     ["info_type_id", "id", "info_type"]],
        "movie_info_idx":[["movie_id", "id", "title"],
                         ["info_type_id", "id", "info_type"]],
        "movie_keyword":[["movie_id", "id", "title"],
                        ["keyword_id", "id", "keyword"]],
        "person_info":[["info_type_id", "id", "info_type"]],
        }
    
    Stats_FKtoKDict = {"badges":[["UserId", "Id", "users"]],
             "comments":[["PostId", "Id", "posts"], ["UserId", "Id", "users"]],
             "postHistory":[["PostId", "Id", "posts"], ["UserId", "Id", "users"]],
             "postLinks":[["PostId", "Id", "posts"]],
             "posts":[["OwnerUserId", "Id", "users"]],
             "tags":[["ExcerptPostId", "Id", "posts"]],
             "votes":[["UserId", "Id", "users"],["PostId", "Id", "posts"]]
            }
    
    queryJGs = None        
    dbConn = None
    if benchmark == 'JOBLight':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightQueries.sql')
        dbConn = getDBConn('imdblight')
    elif benchmark == 'JOBLightRanges':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightRangesQueries.sql')
        dbConn = getDBConn('imdblightranges')
    elif benchmark == 'JOBM':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBMQueries.sql')
        dbConn = getDBConn('imdbm')
    elif benchmark == 'Stats':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/StatsQueries.sql')
        dbConn = getDBConn('stats')
    elif benchmark == 'JOBLight2D':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightQueries.sql')
        dbConn = getDBConn('imdblight2d')
    elif benchmark == 'JOBLightRanges2D':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightRangesQueries.sql')
        dbConn = getDBConn('imdblightranges2d')
    elif benchmark == 'JOBM2D':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBMQueries.sql')
        dbConn = getDBConn('imdbm2d')
    elif benchmark == 'Stats2D':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/StatsQueries.sql')
        dbConn = getDBConn('stats2d')
    elif benchmark == 'JOBLight_FK_PK':
        queryJGs =  [removePKJoins(x, IMDB_FKtoKDict) for x in SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightQueries.sql')]
        dbConn = getDBConn('imdb_fk_pk')
    elif benchmark == 'JOBLightRanges_FK_PK':
        queryJGs =  [removePKJoins(x, IMDB_FKtoKDict) for x in SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightRangesQueries.sql')]
        dbConn = getDBConn('imdb_fk_pk')
    elif benchmark == 'JOBM_FK_PK':
        queryJGs =  [removePKJoins(x, IMDB_FKtoKDict) for x in SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBMQueries.sql')]
        dbConn = getDBConn('imdb_fk_pk')
    elif benchmark == 'Stats_FK_PK':
        queryJGs =  [removePKJoins(x, Stats_FKtoKDict) for x in SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/StatsQueries.sql')]
        dbConn = getDBConn('stats_fk_pk')
    
    dbConn.changeStatisticsTarget(statisticsTarget)
    queryLabels = []
    inferenceTimes = []
    postgresEstimates = []
    for i, queryJG in enumerate(queryJGs):
        if i % 10 == 1:
            print("Postgres Inference Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
        queryLabels.append(i)
        inferenceStart = datetime.now()
        bound = dbConn.getSizeEstimate(queryJG)
        inferenceEnd = datetime.now()
        inferenceTimes.append((inferenceEnd-inferenceStart).total_seconds())
        postgresEstimates.append(bound)
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['InferenceTime'] = inferenceTimes
    inferenceResults['Estimate'] = postgresEstimates
    inferenceResults['StatsSize'] = dbConn.memory()
    inferenceResults.to_csv(outputFile)

    
def evaluate_inference_bayes_card(ensembleDirectory, 
                                  benchmark, 
                                  outputFile):
    sqlFile = None    
    schema = None
    hasTrueCardinality = False
    if benchmark == 'JOBLight':
        csvPath = rootDirectory + "Data/JOB"
        sqlFile =  rootDirectory + 'Workloads/JOBLightQueriesBayes.sql'
        schema = gen_job_light_imdb_schema(csvPath)
        hasTrueCardinality = True
    elif benchmark == 'JOBLightRanges':
        sqlFile =  rootDirectory + 'Workloads/JOBLightRangesQueries.sql'
        return
    elif benchmark == 'JOBM':
        sqlFile =  rootDirectory + 'Workloads/JOBMQueries.sql'
        return
    elif benchmark == 'Stats':
        csvPath = rootDirectory + "Data/Stats/stats_simplified_bayes_card"
        sqlFile =  rootDirectory + 'Workloads/StatsQueriesBayes.sql'
        schema = gen_stats_light_schema(csvPath)
        hasTrueCardinality = True
    
    parsedQueries, true = prepare_join_queries(schema, ensembleDirectory, pairwise_rdc_path=None, 
                                                query_filename=sqlFile, true_card_exist=True)
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
    queries = bn_ensemble.parse_query_all(parsedQueries)
    
    queryLabels = []
    inferenceTimes = []
    estimates = []
    for i, query in enumerate(queries):
        if i % 10 == 1:
            print("BayesCard Inference Exps " + benchmark + " :" + str(100*float(i)/len(queries)) + "% Done")
        queryLabels.append(i)
        inferenceStart = datetime.now()
        estimate = bn_ensemble.cardinality(query)
        if isinstance(estimate, np.ndarray):
            estimate = estimate[0]
        print(type(estimate))
        inferenceEnd = datetime.now()
        inferenceTimes.append((inferenceEnd-inferenceStart).total_seconds())
        estimates.append(estimate)
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['InferenceTime'] = inferenceTimes
    inferenceResults['Estimate'] = estimates
    inferenceResults['StatsSize'] = modelSize
    inferenceResults.to_csv(outputFile)
        
    
def evaluate_inference_pessimistic_cardinality_estimation(benchmark, outputFile):
    queryJGs = None    
    if benchmark == 'JOBLight':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightQueries.sql')
    elif benchmark == 'JOBLightRanges':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightRangesQueries.sql')
    elif benchmark == 'JOBM':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBMQueries.sql')
    elif benchmark == 'Stats':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/StatsQueries.sql')
    
    queryLabels = []
    inferenceTimes = []
    estimates = []
    for i, queryJG in enumerate(queryJGs):
        if i % 10 == 1:
            print("PessEst Inference Exps " + benchmark + " :" + str(100*float(i)/len(queryJGs)) + "% Done")
        queryLabels.append(i)
        queryHints = []
        hintFile = '/home/ec2-user/FrequencyBounds/pqo-opensource/output/results/' + benchmark + "_" + str(i) + "_SubQueryBounds.txt"
        allRelationsEstimate = 0
        maxNumRelations = 0
        for hintString in open(hintFile, "r").readlines():
            relations, estimate = hintString.split("|")
            relations = relations.split(",")
            estimate = int(estimate)
            if len(relations) > maxNumRelations:
                maxNumRelations = len(relations)
                allRelationsEstimate = estimate
        inferenceTimeFile = '/home/ec2-user/FrequencyBounds/pqo-opensource/output/results/' + benchmark + "_" + str(i) + "_sketch_preprocessing.txt"
        inferenceTime = float(open(inferenceTimeFile, "r").readline())
        inferenceTimes.append(inferenceTime)
        estimates.append(allRelationsEstimate)

    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = queryLabels
    inferenceResults['InferenceTime'] = inferenceTimes
    inferenceResults['Estimate'] = estimates
    inferenceResults['StatsSize'] = -1
    inferenceResults.to_csv(outputFile)
    
    

def evaluate_inference_neurocard(benchmark, outputFile):
    if benchmark == "Stats":
        return
    
    estimatesDF = pd.read_csv("/home/ec2-user/FrequencyBounds/Data/Results/NeuroCard_Inference_" + benchmark + ".csv")
    
    statsSize = 0
    if benchmark == "JOBM":
        statsSize = 27.3 * 10**6
    elif benchmark == "JOBLight":
        statsSize = 3.8 * 10**6
    elif benchmark == "JOBLightRanges":
        statsSize = 4.1 * 10**6
    
    inferenceResults = pd.DataFrame()
    inferenceResults['QueryLabel'] = range(0, len(estimatesDF))
    inferenceResults['InferenceTime'] = estimatesDF["InferenceTime"]
    inferenceResults['Estimate'] = estimatesDF["Estimate"]
    inferenceResults['StatsSize'] = statsSize
    inferenceResults.to_csv(outputFile)
    
    
def evaluate_inference(method = 'SafeBound', 
                       statsFile = '../../StatObjects/SafeBound_JOBLight_1.csv',
                       benchmark = 'JOBLight',
                       outputFile = '../../Data/Results/SafeBound_Inference_JOBLight_1.csv',
                       statisticsTarget = None):
    if method == 'SafeBound':
        evaluate_inference_safe_bound(statsFile, benchmark, outputFile)
    if method == 'Simplicity':
        evaluate_inference_simplicity(statsFile, benchmark, outputFile)
    elif method == 'Postgres':
        evaluate_inference_postgres(benchmark, outputFile, statisticsTarget)
    elif method == 'Postgres2D':
        evaluate_inference_postgres(benchmark + "2D", outputFile, statisticsTarget)
    elif method == 'Postgres_FK_PK':
        evaluate_inference_postgres(benchmark + "_FK_PK", outputFile, statisticsTarget)
    elif method == 'BayesCard':
        evaluate_inference_bayes_card(statsFile, benchmark, outputFile)
    elif method == 'PessemisticCardinality':
        evaluate_inference_pessimistic_cardinality_estimation(benchmark, outputFile)