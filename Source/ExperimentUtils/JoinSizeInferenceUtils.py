rootDirectory = '/home/ec2-user/FrequencyBounds/'
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import re
import sys
sys.path.append(rootDirectory + 'Source')
from DBConnectionUtils import *
from SafeBoundUtils import *
from SQLParser import *
sys.path.append(rootDirectory + 'BayesCard')
from Schemas.stats.schema import gen_stats_light_schema
from Schemas.imdb.schema import gen_job_light_imdb_schema    
from DataPrepare.query_prepare_BayesCard import prepare_join_queries
from Models.BN_ensemble_model import BN_ensemble

def reset_cache():
    os.system("sudo sh -c '/usr/bin/echo 3 > /proc/sys/vm/drop_caches'")
    os.system("sudo systemctl restart postgresql-13")

def evaluate_join_size_inference_safe_bound(trueCardFile, 
                                statsFile, 
                                benchmark,
                                queryJGs,
                                outputFile,
                                performTableScan):
    stats = pickle.load(open(statsFile, 'rb'))
    trueCardQueryHints = pickle.load(open(trueCardFile, "rb"))
    
    numTables = []
    relativeErrors = []
    for i, queryJG in enumerate(queryJGs):
        queryHints = []
        for subQuery in queryJG.getIntermediateQueries():
            if performTableScan:
                estimate = stats.functionalFrequencyBound(subQuery, dbConn = dbConn)
            else:
                estimate = stats.functionalFrequencyBound(subQuery)
            tables = list(subQuery.vertexDict.keys())
            hint = JoinHint(tables, estimate)
            queryHints.append(hint)

        trueCardHints = trueCardQueryHints[i]
        for hint in queryHints:
            for trueHint in trueCardHints:
                if sorted(trueHint.tables) == sorted(hint.tables):
                    numTables.append(len(hint.tables))
                    relativeErrors.append(hint.rowEstimate/max(1, trueHint.rowEstimate))
            
    inferenceResults = pd.DataFrame()
    inferenceResults['NumTables'] = numTables
    inferenceResults['RelativeErrors'] = relativeErrors
    inferenceResults.to_csv(outputFile)
    
    
    
def evaluate_join_size_inference_simplicity(trueCardFile, 
                                statsFile, 
                                benchmark,
                                queryJGs,
                                dbConn,
                                outputFile):
    dbConn.changeStatisticsTarget(100)
    stats = pickle.load(open(statsFile, 'rb'))
    
    trueCardQueryHints = pickle.load(open(trueCardFile, "rb"))
    numTables = []
    relativeErrors = []
    for i, queryJG in enumerate(queryJGs):
        queryHints = []
        for subQuery in queryJG.getIntermediateQueries():
            estimate = stats.getSimplicityBound(subQuery, dbConn)
            tables = list(subQuery.vertexDict.keys())
            hint = JoinHint(tables, estimate)
            queryHints.append(hint)

        trueCardHints = trueCardQueryHints[i]
        for hint in queryHints:
            for trueHint in trueCardHints:
                if sorted(trueHint.tables) == sorted(hint.tables):
                    numTables.append(len(hint.tables))
                    relativeErrors.append(hint.rowEstimate/max(1, trueHint.rowEstimate))
            
    inferenceResults = pd.DataFrame()
    inferenceResults['NumTables'] = numTables
    inferenceResults['RelativeErrors'] = relativeErrors
    inferenceResults.to_csv(outputFile)
    
    
def evaluate_join_size_inference_postgres(trueCardFile, 
                              benchmark,
                              queryJGs,
                              dbConn,
                              statisticsTarget,
                              outputFile):
    dbConn.changeStatisticsTarget(statisticsTarget)
    
    trueCardQueryHints = pickle.load(open(trueCardFile, "rb"))
    numTables = []
    relativeErrors = []
    for i, queryJG in enumerate(queryJGs):
        queryHints = []
        for subQuery in queryJG.getIntermediateQueries():
            estimate = dbConn.getSizeEstimate(subQuery)
            tables = list(subQuery.vertexDict.keys())
            hint = JoinHint(tables, estimate)
            queryHints.append(hint)

        trueCardHints = trueCardQueryHints[i]
        for hint in queryHints:
            for trueHint in trueCardHints:
                if sorted(trueHint.tables) == sorted(hint.tables):
                    numTables.append(len(hint.tables))
                    relativeErrors.append(hint.rowEstimate/max(1, trueHint.rowEstimate))
            
    inferenceResults = pd.DataFrame()
    inferenceResults['NumTables'] = numTables
    inferenceResults['RelativeErrors'] = relativeErrors
    inferenceResults.to_csv(outputFile)
    
def evaluate_join_size_inference_postgres2d(trueCardFile, 
                              benchmark,
                              queryJGs,
                              dbConn,
                              statisticsTarget,
                              outputFile):
    dbConn.changeStatisticsTarget(statisticsTarget)
    trueCardQueryHints = pickle.load(open(trueCardFile, "rb"))
    numTables = []
    relativeErrors = []
    for i, queryJG in enumerate(queryJGs):
        queryHints = []
        for subQuery in queryJG.getIntermediateQueries():
            estimate = dbConn.getSizeEstimate(subQuery)
            tables = list(subQuery.vertexDict.keys())
            hint = JoinHint(tables, estimate)
            queryHints.append(hint)

        trueCardHints = trueCardQueryHints[i]
        for hint in queryHints:
            for trueHint in trueCardHints:
                if sorted(trueHint.tables) == sorted(hint.tables):
                    numTables.append(len(hint.tables))
                    relativeErrors.append(hint.rowEstimate/max(1, trueHint.rowEstimate))
            
    inferenceResults = pd.DataFrame()
    inferenceResults['NumTables'] = numTables
    inferenceResults['RelativeErrors'] = relativeErrors
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

def evaluate_join_size_inference_bayes_card(trueCardFile, 
                                ensembleDirectory, 
                                  benchmark,
                                  queryJGs,
                                  outputFile):
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
    
    trueCardQueryHints = pickle.load(open(trueCardFile, "rb"))
    numTables = []
    relativeErrors = []
    for i, queryJG in enumerate(queryJGs):
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
        for subQueryIndex, subQuery in enumerate(subQueries):
            tables = list(subQueryJGs[subQueryIndex].vertexDict.keys())
            estimate = bn_ensemble.cardinality(subQuery)
            hint = JoinHint(tables, estimate)
            queryHints.append(hint)

        trueCardHints = trueCardQueryHints[i]
        for hint in queryHints:
            for trueHint in trueCardHints:
                if sorted(trueHint.tables) == sorted(hint.tables):
                    numTables.append(len(hint.tables))
                    relativeErrors.append(hint.rowEstimate/max(1, trueHint.rowEstimate))
            
    inferenceResults = pd.DataFrame()
    inferenceResults['NumTables'] = numTables
    inferenceResults['RelativeErrors'] = relativeErrors
    inferenceResults.to_csv(outputFile)
    
    
def evaluate_join_size_inference_pessimistic_cardinality_estimation(trueCardFile, 
                                                        statsFile, benchmark, queryJGs, outputFile):
    trueCardQueryHints = pickle.load(open(trueCardFile, "rb"))
    numTables = []
    relativeErrors = []
    for i, queryJG in enumerate(queryJGs):
        queryHints = []
        hintFile = '/home/ec2-user/FrequencyBounds/pqo-opensource/output/results/' + benchmark + "_" + str(i) + "_SubQueryBounds.txt"
        for hintString in open(hintFile, "r").readlines():
            relations, estimate = hintString.split("|")
            relations = relations.split(",")
            estimate = int(estimate)
            hint = JoinHint(relations, estimate)
            queryHints.append(hint)
        trueCardHints = trueCardQueryHints[i]
        for hint in queryHints:
            hint.tables = [x.upper() for x in hint.tables]
            for trueHint in trueCardHints:
                if sorted(trueHint.tables) == sorted(hint.tables):
                    numTables.append(len(hint.tables))
                    relativeErrors.append(hint.rowEstimate/max(1, trueHint.rowEstimate))
            
    inferenceResults = pd.DataFrame()
    inferenceResults['NumTables'] = numTables
    inferenceResults['RelativeErrors'] = relativeErrors
    inferenceResults.to_csv(outputFile)
    
def evaluate_join_size_inference_neurocard(trueCardFile,  benchmark, queryJGs, outputFile):
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
    
    trueCardQueryHints = pickle.load(open(trueCardFile, "rb"))
    
    numTables = []
    relativeErrors = []
    for i, queryJG in enumerate(queryJGs):
        queryHints = allQueryHints[i]
        trueCardHints = trueCardQueryHints[i]
        for hint in queryHints:
            for trueHint in trueCardHints:
                if sorted(trueHint.tables) == sorted(hint.tables):
                    numTables.append(len(hint.tables))
                    relativeErrors.append(hint.rowEstimate/max(1, trueHint.rowEstimate))
            
    
    inferenceResults = pd.DataFrame()
    inferenceResults['NumTables'] = numTables
    inferenceResults['RelativeErrors'] = relativeErrors
    inferenceResults.to_csv(outputFile)
    
def evaluate_join_size_inference(method = 'SafeBound', 
                    statsFile = '../../StatObjects/SafeBound_JOBLight_1.csv',
                    benchmark = 'JOBLight',
                    outputFile = '../../Data/Results/SafeBound_Inference_JOBLight_1.csv',
                    statisticsTarget = None):
    
    queryJGs = None        
    dbConn = None
    if benchmark == 'JOBLight':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightQueries.sql')
        if method == 'Postgres2D':
            dbConn = getDBConn('imdblight2d')
        else:
            dbConn = getDBConn('imdblight')
    elif benchmark == 'JOBLightRanges':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBLightRangesQueries.sql')
        if method == 'Postgres2D':
            dbConn = getDBConn('imdblightranges2d')
        elif method == 'Postgres_FK_PK':
            dbConn = getDBConn('imdblightranges')
        else:
            dbConn = getDBConn('imdblightranges')
    elif benchmark == 'JOBM':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/JOBMQueries.sql')
        if method == 'Postgres2D':
            dbConn = getDBConn('imdbm2d')
        else:
            dbConn = getDBConn('imdbm')
    elif benchmark == 'Stats':
        queryJGs =  SQLFileToJoinQueryGraphs(rootDirectory + 'Workloads/StatsQueries.sql')
        if method == 'Postgres2D':
            dbConn = getDBConn('stats2d')
        else:
            dbConn = getDBConn('stats')
    
    trueCardFile = rootDirectory + "StatObjects/TrueCardinality_" + benchmark + ".pkl" 
    
    if method == 'SafeBound':
        evaluate_join_size_inference_safe_bound(trueCardFile, statsFile, benchmark, queryJGs, outputFile, performTableScan = False)
    elif method == "SafeBoundScan":
        evaluate_join_size_inference_safe_bound(trueCardFile, statsFile, benchmark, queryJGs, outputFile, performTableScan = True)
    elif method == "Simplicity":
        evaluate_join_size_inference_simplicity(trueCardFile, statsFile, benchmark, queryJGs, dbConn, outputFile)
    elif method == 'Postgres':
        evaluate_join_size_inference_postgres(trueCardFile, benchmark, queryJGs, dbConn, statisticsTarget, outputFile)
    elif method == 'Postgres2D':
        evaluate_join_size_inference_postgres(trueCardFile, benchmark, queryJGs, dbConn, statisticsTarget, outputFile)
    elif method == 'BayesCard':
        evaluate_join_size_inference_bayes_card(trueCardFile, statsFile, benchmark, queryJGs, outputFile)
    elif method == 'PessemisticCardinality':
        evaluate_join_size_inference_pessimistic_cardinality_estimation(trueCardFile, statsFile, benchmark, queryJGs, outputFile)
    elif method == 'NeuroCard':
        evaluate_join_size_inference_neurocard(trueCardFile, benchmark, queryJGs, outputFile)