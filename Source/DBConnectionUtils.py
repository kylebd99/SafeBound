import re
import psycopg2
import numpy as np
import sys, os
rootFileDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +'/'
sys.path.append(rootFileDirectory + 'Source')
from JoinGraphUtils import *

def getDBConn(dbName="IMDB"):
    conn = psycopg2.connect(dbname=dbName, host='/var/run/postgresql', port='5432' )
    conn.set_session(autocommit = True)    
    conn.cursor().execute("Load 'pg_hint_plan';")
    dbConn = DatabaseConnection(conn, dbName)
    return dbConn

class DatabaseConnection:
    
    def __init__(self, pgConnection, dbName):
        self.conn = pgConnection
        self.dbName = dbName
        
    def reset(self):
        self.conn.close()
        self.conn = getDBConn(self.dbName).conn
        
    def getDegreeSequence(self, tableName, joinColumn, predicates):
        query = "SELECT COUNT(*) as c FROM "
        query += tableName 
        if len(predicates) > 0:
            query += " WHERE "
            prefix = ""
            for predicate in predicates:
                columnName = re.sub(r'.*\.', '', predicate.colName, count=1)
                fixedPred = Predicate(columnName, predicate.predType, predicate.compValue)
                query += prefix
                query += tableName + "." + fixedPred.toString()
                prefix = " AND "
        query += " GROUP BY " + tableName + "." + re.sub(r'.*\.', '', joinColumn, count=1)
        query += " ORDER BY c DESC;"
        cursor = self.conn.cursor()
        cursor.execute(query)
        degreeSequence = []
        for degree in cursor:
            degreeSequence.append(degree[0])
        degreeSequence = [int(degree) for degree in degreeSequence]
        print(min(degreeSequence))
        print(max(degreeSequence))
        return np.array(degreeSequence)
        
    def getSizeEstimate(self, joinQueryGraph):
        query = joinQueryGraph.getSQLQuery()
        cursor = self.conn.cursor()
        query = "EXPLAIN " + query
        cursor.execute(query)
        results = cursor.fetchone()[0]
        estimate = int(re.findall(r"rows=(\d*)", results)[0])
        return estimate
        
    def getSizeEstimateAndActual(self, joinQueryGraph, hints = []):
        cursor = self.conn.cursor()
        query = "/*+\n"
        for hint in hints:
            query += "Rows("
            for table in hint.tables:
                query += table.lower() + " "
            query += "#" + str(min(8223372036854775800, hint.rowEstimate)) + ")\n"
        query += "*/\n"
        explainQuery = query
        explainQuery += "EXPLAIN " + joinQueryGraph.getSQLQuery(countStar=False)
        cursor.execute(explainQuery)
        results = cursor.fetchone()[0]
        estimate = int(re.findall(r"rows=(\d*)", results)[0])
        
        query += joinQueryGraph.getSQLQuery(countStar=True)
        cursor.execute(query)
        actual = cursor.fetchone()[0]
        actualInt = int(actual)
        return estimate, actual
        
    def createStatistics(self, leftCol, rightCol, tableName):
        cursor = self.conn.cursor()
        query = "CREATE STATISTICS " + tableName + "_" + leftCol + "_" + rightCol + " ON " + leftCol + ", " + rightCol + " FROM " + tableName + ";"
        cursor.execute(query)
    
    def dropStatistics(self, leftCol, rightCol, tableName):
        cursor =self.conn.cursor()
        query = "DROP STATISTICS IF EXISTS " + tableName + "_" + leftCol + "_" + rightCol + ";"
        cursor.execute(query)

    def runAnalyze(self):
        cursor =self.conn.cursor()
        query = "ANALYZE;"
        cursor.execute(query)
        query = "VACUUM FULL pg_statistic;"
        cursor.execute(query)
        query = "VACUUM FULL pg_statistic_ext_data;"
        cursor.execute(query)
        

    def printQueryPlan(self, joinQueryGraph, hints=[], analyze=False):
        cursor = self.conn.cursor()
        query = "/*+\n"
        for hint in hints:
            query += "Rows("
            for table in hint.tables:
                query += table.lower() + " "
            query += "#" + str(hint.rowEstimate) + ")\n"
        query += "*/\n"
        query += "EXPLAIN "
        if analyze:
            query += "ANALYZE "
        query += joinQueryGraph.getSQLQuery(countStar=True)
        cursor.execute(query)
        results = cursor.fetchall()
        for l in results:
            print(l)    
    def printQuery(self, joinQueryGraph, hints=[]):
        query = "/*+\n"
        for hint in hints:
            query += "Rows("
            for table in hint.tables:
                query += table.lower() + " "
            query += "#" + str(hint.rowEstimate) + ")\n"
        query += "*/\n"
        query += "EXPLAIN " + joinQueryGraph.getSQLQuery()
        print(query)
        
    def changeStatisticsTarget(self, target):
        query = "ALTER SYSTEM SET default_statistics_target=" + str(target) + ";"
        cursor = self.conn.cursor()
        cursor.execute(query)
        query = "SELECT pg_reload_conf();"
        cursor.execute(query)
        self.runAnalyze()
    
    def memory(self):
        query = "SELECT pg_total_relation_size('pg_statistic')+pg_total_relation_size('pg_statistic_ext_data');"
        cursor = self.conn.cursor()
        cursor.execute(query)
        memory = cursor.fetchone()[0]
        return int(memory)
        
    def close(self):
        self.conn.close()
