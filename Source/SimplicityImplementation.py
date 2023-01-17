from JoinGraphUtils import *
from DBConnectionUtils import *
from ExperimentUtils.LoadUtils import *


class Simplicity:
    def __init__(self, tableDFs, tableNames, joinCols):
        self.tableMFDict = dict()
        self.tableCardDict = dict()
        self.tableNames = [x.upper() for x in tableNames]
        self.joinCols = [[x.upper() for x in tableJoinCols] for tableJoinCols in joinCols]
        
        
        for i in range(len(tableDFs)):
            tableDF = tableDFs[i]
            tableDF.columns = [x.upper() for x in tableDF.columns]
            
            self.tableCardDict[self.tableNames[i]] = len(tableDF)
            self.tableMFDict[self.tableNames[i]] = dict()
            for joinCol in self.joinCols[i]:
                self.tableMFDict[self.tableNames[i]][joinCol] = tableDF.groupby(joinCol).size().max()
        
    def getSingleTableCardinality(self, tableName, predicates, dbConn):
        return dbConn.getSingleTableCardinalityEstimate(tableName, predicates)
        
    def getSimplicityBound(self, joinQueryGraph, dbConn):
        bound = 2**63
        for alias in joinQueryGraph.tableDict.keys():
            graphIter = JoinQueryGraphIter(joinQueryGraph, alias)
            graphIter.getNext()
            tableName = joinQueryGraph.tableDict[alias]
            tempBound = self.getSingleTableCardinality(tableName, joinQueryGraph.vertexDict[alias].predicates, dbConn)
            while graphIter.hasNext():
                nextJoin = graphIter.getNext()
                nextAlias = nextJoin[2]
                nextTable = joinQueryGraph.tableDict[nextAlias]
                nextJoinCol = nextJoin[3]
                nextMF = self.tableMFDict[nextTable][nextJoinCol]
                tempBound *= nextMF
            bound = min(bound, tempBound)
        return bound
    
    def memory(self):
        memory = self.tableCardDict.__sizeof__()
        for tableName in self.tableNames:
            memory += self.tableMFDict[tableName].__sizeof__()
        return memory
    
    