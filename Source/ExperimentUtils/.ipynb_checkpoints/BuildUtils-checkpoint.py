import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import sys
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +'/'
sys.path.append(rootFileDirectory + 'Source')
sys.path.append(rootFileDirectory + 'Source/ExperimentUtils')
from SafeBoundUtils import *
from SimplicityImplementation import *
from DBConnectionUtils import *
from LoadUtils import *
sys.path.append(rootFileDirectory + 'bayescard')
from Schemas.stats.schema import gen_stats_light_schema
from Schemas.imdb.schema import gen_job_light_imdb_schema
from DataPrepare.join_data_preparation import JoinDataPreparator
from Models.Bayescard_BN import Bayescard_BN, build_meta_info
from DeepDBUtils.evaluation.utils import timestamp_transorform


def build_safe_bound(benchmark, parameters, outputFile):
    tables = None
    tableNames = None
    joinColumns = None
    filterColumns = None
    FKtoKDict = None
    relativeErrorPerSegment = parameters['relativeErrorPerSegment']
    numHistogramBuckets = parameters['numHistogramBuckets']
    numEqualityOutliers = parameters['numEqualityOutliers']
    numCDFGroups = parameters['numCDFGroups']
    trackNulls = parameters['trackNulls']
    trackTriGrams = parameters['trackTriGrams']
    numCores = parameters['numCores']
    verbose = parameters['verbose']
    groupingMethod = parameters["groupingMethod"]
    modelCDF = parameters["modelCDF"]
    
    
    if benchmark == 'JOBLight':
        data = load_imdb()
        
        tableNames = ["cast_info", 
                      "movie_companies",
                      "movie_info_idx",
                      "movie_info",
                      "movie_keyword",
                      "title"]

        joinColumns = [["movie_id"],
                             ["movie_id"],
                             ["movie_id"],
                             ["movie_id"],
                             ["movie_id"],
                             ["id", "kind_id"]
                            ]
        
        filterColumns = [["role_id", "nr_order"],
                               ["company_type_id", 'company_id'],
                               ["info_type_id"],
                               ["info_type_id"],
                               ["keyword_id"],
                               ["episode_nr", "season_nr", "production_year", "kind_id"]
                              ]
        
        tables = [data[table][list(set(joinColumns[i] + filterColumns[i]))] for i, table in enumerate(tableNames)]
        del data
        
        FKtoKDict = {"cast_info":[["movie_id", "id", "title"]],
                    "movie_companies":[["movie_id", "id", "title"]],
                    "movie_info":[["movie_id", "id", "title"]],
                    "movie_info_idx":[["movie_id", "id", "title"]],
                    "movie_keyword":[["movie_id", "id", "title"]]
                    }
        
    elif benchmark == 'JOBLightRanges':
        data = load_imdb()    

        tableNames = ["cast_info", "movie_companies", "movie_info_idx", "movie_info",
                     "movie_keyword", "title"]
                  
        joinColumns = [ ["movie_id"],
                        ["movie_id"],
                        ["movie_id"],
                        ["movie_id"],
                        ["movie_id"],
                        ["id", "kind_id"]
                        ]
        filterColumns = [
                        ["role_id", "nr_order"],
                        ["company_type_id", 'company_id'],
                        ["info_type_id"],
                        ["info_type_id"],
                        ["keyword_id"],
                        ["episode_nr", "season_nr", "production_year", "series_years", "kind_id", 'phonetic_code', 'series_years', 'imdb_index']
                        ]
        tables = [data[table][list(set(joinColumns[i] + filterColumns[i]))] for i, table in enumerate(tableNames)]
        del data
        
        FKtoKDict = {"cast_info":[["movie_id", "id", "title"]],
                    "movie_companies":[["movie_id", "id", "title"]],
                    "movie_info":[["movie_id", "id", "title"]],
                    "movie_info_idx":[["movie_id", "id", "title"]],
                    "movie_keyword":[["movie_id", "id", "title"]]
                    }
        
    elif benchmark == 'JOBM': 
        data = load_imdb()

        tableNames = [ "cast_info", "aka_name", "aka_title",
                          "comp_cast_type", "company_name",
                     "company_type", "complete_cast", "info_type",
                     "keyword", "kind_type", "link_type",
                     "movie_companies", "movie_info_idx", "movie_info",
                     "movie_keyword", "movie_link", "role_type", "title"]
        
        joinColumns = [
                           ["id", "person_id", "movie_id", "person_role_id", "role_id"],
                           ["id", "person_id"],
                           ["id", "movie_id", "kind_id"],
                           ["id"],
                           ["id"],
                           ["id"],
                           ["id", "movie_id", "subject_id", "status_id"],
                           ["id"],
                           ["id"],
                           ["id"],
                           ["id"],
                           ["id", "movie_id", "company_id", "company_type_id"],
                           ["id", "movie_id", "info_type_id"],
                           ["id", "movie_id", "info_type_id"],
                           ["id", "movie_id", "keyword_id"],
                           ["id", "movie_id", "linked_movie_id", "link_type_id"],
                           ["id"],
                           ["id", "kind_id"]
                          ]

        filterColumns = [["note"],
                           [],
                           [],
                           ["kind"],
                           ["country_code", "name"],
                           ["kind"],
                           [],
                           ["info"],
                           ["keyword"],
                           ["kind"],
                           ["link"],
                           ["note"],
                           ["info"],
                           ["info", "note"],
                           [],
                           [],
                           [],
                           ["episode_nr", "production_year", "title"]
                        ]
                
        tables = [data[table][list(set(joinColumns[i] + filterColumns[i]))] for i, table in enumerate(tableNames)]
        del data
        
        FKtoKDict = {"aka_title":[["movie_id", "id", "title"], ["kind_id", "id", "kind_type"]],
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
        
    elif benchmark == 'Stats':
        data = load_stats()
        
        tableNames = ["badges",
                      "comments", 
                      "postHistory",
                      "postLinks", 
                      "posts", 
                      "tags", 
                      "users", 
                      "votes"]
        
        joinColumns = [["Id","UserId"], 
                        ["Id", "PostId", "UserId"],
                        ["Id", "PostId", "UserId"],
                        ["Id", "PostId", "RelatedPostId"],
                        ["Id", "OwnerUserId", "LastEditorUserId"],
                        ["Id", "ExcerptPostId"],
                        ["Id"],
                        ["Id", "PostId", "UserId"]]
        filterColumns = [["Date"],
                          ["Score", "CreationDate"],
                          ["PostHistoryTypeId", "CreationDate"],
                          ["CreationDate", "LinkTypeId"],
                          ["PostTypeId", "CreationDate", "Score", "ViewCount", "AnswerCount", "CommentCount", "FavoriteCount"],
                          ["Count"],
                          ["Reputation", "CreationDate", "Views", "UpVotes", "DownVotes"],
                          ["VoteTypeId", "CreationDate", "BountyAmount"]
                        ]
                
        
        FKtoKDict = {"badges":[["UserId", "Id", "users"]],
                     "comments":[["PostId", "Id", "posts"], ["UserId", "Id", "users"]],
                     "postHistory":[["PostId", "Id", "posts"], ["UserId", "Id", "users"]],
                     "postLinks":[["PostId", "Id", "posts"]],
                     "posts":[["OwnerUserId", "Id", "users"]],
                     "tags":[["ExcerptPostId", "Id", "posts"]],
                     "votes":[["UserId", "Id", "users"],["PostId", "Id", "posts"]]
                    }
        tables = [data[table][list(set(joinColumns[i] + filterColumns[i]))] for i, table in enumerate(tableNames)]
        del data
        
    elif "TPCH" in benchmark:
        size = benchmark.split("-")[1]
        
        data = load_tpch(size)
        
        tableNames = ["nation", "region", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
        joinColumns = [["n_nationkey", "n_regionkey"], 
                       ["r_regionkey"],
                       ["s_suppkey", "s_nationkey"],
                       ["c_custkey", "c_nationkey"],
                       ["p_partkey"],
                       ["ps_partkey", "ps_suppkey"],
                       ["o_orderkey", "o_custkey"],
                       ["l_orderkey", "l_partkey", "l_suppkey"]]

        filterColumns = [["n_name", "n_comment"], 
                         ["r_name", "r_comment"],
                         ["s_name", "s_address", "s_phone", "s_acctbal", "s_comment"],
                         ["c_name", "c_address", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"],
                         ["p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"],
                         ["ps_availqty", "ps_supplycost", "ps_comment"],
                         ["o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
                         ["l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate",
                              "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]]

        FKtoKDict = {"nation":[["n_regionkey", "r_regionkey", "region"]],
                     "supplier":[["s_nationkey", "n_nationkey", "nation"]],
                     "customer":[["c_nationkey", "n_nationkey", "nation"]],
                     "partsupp":[["ps_partkey", "p_partkey", "part"],
                                 ["ps_suppkey", "s_suppkey", 'supplier']],
                     "orders":[["o_custkey", "c_custkey", "customer"]],
                     "lineitem":[["l_orderkey", "o_orderkey", "orders"],
                         ["l_partkey", "p_partkey", "part"],
                         ["l_suppkey", "s_suppkey", "supplier"]]}
        
        tables = [data[table][list(set(joinColumns[i] + filterColumns[i]))] for i, table in enumerate(tableNames)]
        for i, table in enumerate(tables):
            for column in table.columns:
                if "key" in column:
                    table.loc[column] = table[column].astype("Int64")

        del data
    buildStart = datetime.now()
    stats = SafeBound(tables, tableNames, joinColumns,
                      relativeErrorPerSegment, filterColumns, numHistogramBuckets, 
                      numEqualityOutliers, FKtoKDict, numCDFGroups,
                      trackNulls, trackTriGrams, numCores, groupingMethod, modelCDF,
                      verbose)
    buildEnd = datetime.now()
    buildSeconds = (buildEnd-buildStart).total_seconds()
    pickle.dump(stats, open(outputFile, "wb" ))
    return buildSeconds, stats.memory()

def build_simplicity(benchmark, parameters, outputFile):
    tables = None
    tableNames = None
    joinColumns = None
    
    
    if benchmark == 'JOBLight':
        data = load_imdb()
        
        tableNames = ["cast_info", 
                      "movie_companies",
                      "movie_info_idx",
                      "movie_info",
                      "movie_keyword",
                      "title"]

        joinColumns = [["movie_id"],
                             ["movie_id"],
                             ["movie_id"],
                             ["movie_id"],
                             ["movie_id"],
                             ["id", "kind_id"]
                            ]
        
        tables = [data[table][list(set(joinColumns[i]))] for i, table in enumerate(tableNames)]
        del data
        
    elif benchmark == 'JOBLightRanges':
        data = load_imdb()    

        tableNames = ["cast_info", "movie_companies", "movie_info_idx", "movie_info",
                     "movie_keyword", "title"]
                  
        joinColumns = [ ["movie_id"],
                        ["movie_id"],
                        ["movie_id"],
                        ["movie_id"],
                        ["movie_id"],
                        ["id", "kind_id"]
                        ]
        tables = [data[table][list(set(joinColumns[i]))] for i, table in enumerate(tableNames)]
        del data
        
    elif benchmark == 'JOBM': 
        data = load_imdb()

        tableNames = [ "cast_info", "aka_name", "aka_title",
                          "comp_cast_type", "company_name",
                     "company_type", "complete_cast", "info_type",
                     "keyword", "kind_type", "link_type",
                     "movie_companies", "movie_info_idx", "movie_info",
                     "movie_keyword", "movie_link", "role_type", "title"]
        
        joinColumns = [
                           ["id", "person_id", "movie_id", "person_role_id", "role_id"],
                           ["id", "person_id"],
                           ["id", "movie_id", "kind_id"],
                           ["id"],
                           ["id"],
                           ["id"],
                           ["id", "movie_id", "subject_id", "status_id"],
                           ["id"],
                           ["id"],
                           ["id"],
                           ["id"],
                           ["id", "movie_id", "company_id", "company_type_id"],
                           ["id", "movie_id", "info_type_id"],
                           ["id", "movie_id", "info_type_id"],
                           ["id", "movie_id", "keyword_id"],
                           ["id", "movie_id", "linked_movie_id", "link_type_id"],
                           ["id"],
                           ["id", "kind_id"]
                          ]

        tables = [data[table][list(set(joinColumns[i]))] for i, table in enumerate(tableNames)]
        del data
        
    elif benchmark == 'Stats':
        data = load_stats()
        
        tableNames = ["badges",
                      "comments", 
                      "postHistory",
                      "postLinks", 
                      "posts", 
                      "tags", 
                      "users", 
                      "votes"]
        
        joinColumns = [["Id","UserId"], 
                        ["Id", "PostId", "UserId"],
                        ["Id", "PostId", "UserId"],
                        ["Id", "PostId", "RelatedPostId"],
                        ["Id", "OwnerUserId", "LastEditorUserId"],
                        ["Id", "ExcerptPostId"],
                        ["Id"],
                        ["Id", "PostId", "UserId"]]
        
        tables = [data[table][list(set(joinColumns[i]))] for i, table in enumerate(tableNames)]
        del data
      
        
    buildStart = datetime.now()
    stats = Simplicity(tables, tableNames, joinColumns)
    buildEnd = datetime.now()
    buildSeconds = (buildEnd-buildStart).total_seconds()
    pickle.dump(stats, open(outputFile, "wb" ))
    return buildSeconds, stats.memory()

def build_postgres(benchmark, parameters):
    
    statisticsTarget = parameters['statisticsTarget']
    dbConn = None
    if benchmark == 'JOBLight':
        dbConn = getDBConn('imdblight')
    elif benchmark == 'JOBLightRanges':
        dbConn = getDBConn('imdblightranges')
    elif benchmark == 'JOBM':
        dbConn = getDBConn('imdbm')
    elif benchmark == 'Stats':
        dbConn = getDBConn('stats')
    elif benchmark == 'JOBLight2D':
        dbConn = getDBConn('imdblight2d')
    elif benchmark == 'JOBLightRanges2D':
        dbConn = getDBConn('imdblightranges2d')
    elif benchmark == 'JOBM2D':
        dbConn = getDBConn('imdbm2d')
    elif benchmark == 'Stats2D':
        dbConn = getDBConn('stats2d')
    
    dbConn.changeStatisticsTarget(statisticsTarget)
    buildStart = datetime.now()
    dbConn.runAnalyze()
    buildEnd = datetime.now()
    return (buildEnd-buildStart).total_seconds(), dbConn.memory()

def build_bayes_card(benchmark, parameters, outputFolder):
    
    hdfPath = None
    csvPath = None
    schema = None
    
    if benchmark == 'Stats':
        hdfPath = rootDirectory + "Data/Stats/stats_simplified_bayes_card/stats_hdf"
        csvPath = rootDirectory + "Data/Stats/stats_simplified_bayes_card"
        schema = gen_stats_light_schema(csvPath)
    elif benchmark == 'JOBLight':
        hdfPath = rootDirectory + "Data/IMDB/JOB_hdf"
        csvPath = rootDirectory + "Data/IMDB"
        schema = gen_job_light_imdb_schema(csvPath)
    else:
        return 0,0
        
    buildStart = datetime.now()
    metaDataPath = hdfPath + '/meta_data.pkl'
    prep = JoinDataPreparator(metaDataPath, schema, max_table_data=20000000)
    
    algorithm = "chow-liu"
    maxParents = 1
    sampleSize = 200000
    modelSize = 0
    for i, relationship_obj in enumerate(schema.relationships):
        dfSampleSize = 10000000
        relation = [relationship_obj.identifier]
        df, metaTypes, nullValues, fullJoinEst = prep.generate_n_samples(dfSampleSize,
                                                                        relationship_list=relation,
                                                                        post_sampling_factor=10)
        columns = list(df.columns)
        assert len(columns) == len(metaTypes) == len(nullValues)
        metaInfo = build_meta_info(df.columns, nullValues)
        bn = Bayescard_BN(schema, relation, column_names=columns, full_join_size=len(df),
                          table_meta_data=prep.table_meta_data, meta_types=metaTypes, null_values=nullValues,
                          meta_info=metaInfo)
        modelPath = outputFolder + f"/{i}_{algorithm}_{maxParents}.pkl"
        bn.build_from_data(df, algorithm=algorithm, max_parents=maxParents, ignore_cols=['Id'],
                           sample_size=sampleSize)
        pickle.dump(bn, open(modelPath, 'wb'), pickle.HIGHEST_PROTOCOL)
        modelSize +=  os.path.getsize(modelPath)
    buildEnd = datetime.now()
    return (buildEnd-buildStart).total_seconds(), modelSize

    
def build_stats_object(method = 'SafeBound',
                       benchmark = 'JOBLight',
                       parameters = dict(), 
                       outputFile = "../../StatObjects/SafeBound_JOBLight.pkl"):    

    if method == 'SafeBound':
        return build_safe_bound(benchmark, parameters, outputFile)
    
    if method == 'Simplicity':
        return build_simplicity(benchmark, parameters, outputFile)
    
    elif method == 'Postgres':
        return build_postgres(benchmark, parameters)

    elif method == 'Postgres2D':
        return build_postgres(benchmark + "2D", parameters)
    
    elif method == 'BayesCard':
        return build_bayes_card(benchmark, parameters, outputFile)
    
    elif method == 'PessimisticCardinality':
        return -1
    
    else:
        return -1
    


    
    


