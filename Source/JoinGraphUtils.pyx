# cython: infer_types=True
import pandas as pd
import numpy as np
import random
import math
from collections import deque
import copy
import networkx as nx
import itertools

# A struct containing information on a particular predicate. Supported predicates include >, <, =, and regex contains
class Predicate:
    def __init__(self, colName, predType, compValue):
        self.colName = colName
        self.predType = predType
        self.compValue = compValue
        
    def toString(self):
        if isinstance(self.compValue, str):
            return self.colName + " " + str(self.predType) + " " + "'" + str(self.compValue) + "'"
        elif isinstance(self.compValue, list):
            string = self.colName + " " + str(self.predType) + " ("
            if isinstance(self.compValue[0], str):
                for value in self.compValue:
                    string += "'" + value + "', "
                string = string[0:len(string)-2]
                string += ")"
                return string
            else:
                for value in self.compValue:
                    string += str(value) + ", "
                string = string[0:len(string)-2]
                string += ")"
                return string
        elif self.predType == "IS NULL" or self.predType == "IS NOT NULL":
            return self.colName + " " + self.predType
        else:
            return self.colName + " " + self.predType + " " + str(self.compValue)


# A struct containing information on a particular table in a join. This includes the next tables to be joined
# and the single table selectivity of any predicates.
class JoinVertex:
    def __init__(self, alias, tableName, childAliases, childJoinCols, outputJoinCols, predicates, numSelectedRows):
        self.alias = alias
        self.tableName = tableName
        self.edgeAliases = childAliases
        self.edgeJoinCols = childJoinCols
        self.outputJoinCols = outputJoinCols
        self.predicates = predicates
        self.numSelectedRows = numSelectedRows

    def addColumnSuffixes(self, tableDict):
        for i in range(len(self.edgeJoinCols)):
            self.edgeJoinCols[i] = tableDict[self.edgeAliases[i]] + "." + self.edgeJoinCols[i]
        for i in range(len(self.outputJoinCols)):
            self.outputJoinCols[i] = self.tableName + "." + self.outputJoinCols[i] 
        for i in range(len(self.predicates)):
            self.predicates[i].colName = self.tableName + "." + self.predicates[i].colName
        
    def copy(self):
        copyPreds = []
        for pred in self.predicates:
            copyPreds.append(Predicate(pred.colName, pred.predType, pred.compValue))
        return JoinVertex(self.alias, self.tableName, self.edgeAliases.copy(),
                          self.edgeJoinCols.copy(), self.outputJoinCols.copy(), copyPreds, self.numSelectedRows)
        
# A struct used to define a join query and the single-table selectivity of predicates.
# All tables must have an alias, lack of selectivity estimate is denoted as -1, and no cycles are allowed.
class JoinQueryGraph:
    
    def __init__(self):
        self.edgeDict = dict()
        self.predDict = dict()
        self.numSelectedRowsDict = dict()
        self.tableDict = dict()
        self.vertexDict = dict()
        self.equalityClasses = []
        self.equalityDict = dict()
        self.hasSuffixes = False
        
    def addAlias(self, table, alias):
        self.tableDict[alias.upper()] = table.upper()
    
    def addJoin(self, leftAlias, leftJoinCol, rightAlias, rightJoinCol):
        leftAlias= leftAlias.upper()
        leftJoinCol = leftJoinCol.upper()
        rightAlias = rightAlias.upper()
        rightJoinCol = rightJoinCol.upper()
        
        if leftAlias not in self.tableDict:
            self.tableDict[leftAlias] = leftAlias
        if rightAlias not in self.tableDict:
            self.tableDict[rightAlias] = rightAlias
            
        if leftAlias not in self.edgeDict:
            self.edgeDict[leftAlias] = set()
        if rightAlias not in self.edgeDict:
            self.edgeDict[rightAlias] = set()
        
        self.edgeDict[leftAlias].add((rightAlias, rightJoinCol, leftJoinCol))
        self.edgeDict[rightAlias].add((leftAlias, leftJoinCol, rightJoinCol))
        
        rightEqualityClass = None
        leftEqualityClass = None
        leftAliasAndCol = leftAlias+"."+self.tableDict[leftAlias] + "."+leftJoinCol
        rightAliasAndCol =  rightAlias+"."+ self.tableDict[rightAlias] + "." + rightJoinCol
        for classId, equalityClass in enumerate(self.equalityClasses):
            for aliasAndCol in equalityClass:
                if leftAliasAndCol == aliasAndCol:
                    leftEqualityClass = classId
                if rightAliasAndCol == aliasAndCol:
                    rightEqualityClass = classId
        if rightEqualityClass is None and leftEqualityClass is None:
            self.equalityClasses.append([leftAliasAndCol, rightAliasAndCol])
        elif rightEqualityClass is None:
            self.equalityClasses[leftEqualityClass].append(rightAliasAndCol)
        elif leftEqualityClass is None:
            self.equalityClasses[rightEqualityClass].append(leftAliasAndCol)
        elif leftEqualityClass != rightEqualityClass:
            self.equalityClasses[leftEqualityClass].extend(self.equalityClasses[rightEqualityClass])
            del self.equalityClasses[rightEqualityClass]        
        
        
    def addSelectivityEstimate(self, alias, numSelectedRows):
        self.numSelectedRowsDict[alias] = numSelectedRows
    
    def addPredicate(self, alias, colName, predType, compValue):
        alias = alias.upper()
        colName = colName.upper()
        if alias not in self.predDict:
            self.predDict[alias] = []
        self.predDict[alias].append(Predicate(colName, predType, compValue))

    # Builds the tree structure of the join graph and upper cases all names.
    def buildJoinGraph(self, addImpliedJoins = True):
        self.vertexDict = dict()
        if addImpliedJoins:
            joinsToAdd = []
            for alias1, edges1 in self.edgeDict.items():
                for alias2, edges2 in self.edgeDict.items():
                    if alias1 == alias2:
                        continue
                    for edge1 in edges1:
                        for edge2 in edges2:
                            if edge1[0] == edge2[0] and edge1[1] == edge2[1]:
                                joinsToAdd.append([alias1, edge1[2], alias2, edge2[2]])
            for join in joinsToAdd:
                self.addJoin(join[0], join[1], join[2], join[3])
            
        for equalityClass, members in enumerate(self.equalityClasses):
            for member in members:
                self.equalityDict[member] = equalityClass
            
        for alias, edges in self.edgeDict.items():
            alias = alias.upper()
            edgeAliases = []
            edgeJoinCols = []
            outputJoinCols = []
            for e in edges:
                edgeAliases.append(e[0].upper())
                edgeJoinCols.append(e[1].upper())
                outputJoinCols.append(e[2].upper())
            table = self.tableDict[alias]
            if alias not in self.predDict:
                self.predDict[alias] = []
            predicates = self.predDict[alias]
            if alias not in self.numSelectedRowsDict:
                self.numSelectedRowsDict[alias] = -1
            numSelectedRows = self.numSelectedRowsDict[alias]
            self.vertexDict[alias] = JoinVertex(alias, table, edgeAliases, edgeJoinCols, outputJoinCols, predicates, numSelectedRows)
    
    def findCycles(self):
        aliasSet = set()
        jIter = JoinQueryGraphIter(self)
        while jIter.hasNext():
            curJoin = jIter.getNext()
            curVertex = self.vertexDict[curJoin[2]]
            aliasSet.add(curVertex.alias)
            for c in curVertex.edgeAliases:
                if (c != curJoin[0]) and (c in aliasSet):
                    return [curVertex.alias, c]
        return None
    
    def arbitrarilyBreakCycles(self):
        circularEdge = self.findCycles()
        while circularEdge != None:
            leftAlias = circularEdge[0]
            leftVertex = self.vertexDict[leftAlias]
            rightAlias = circularEdge[1]
            rightVertex = self.vertexDict[rightAlias]
            for i in range(len(leftVertex.edgeAliases)):
                if leftVertex.edgeAliases[i] == rightAlias:
                    del leftVertex.edgeAliases[i]
                    del leftVertex.edgeJoinCols[i]
                    del leftVertex.outputJoinCols[i]
                    break
            for i in range(len(rightVertex.edgeAliases)):
                if rightVertex.edgeAliases[i] == leftAlias:
                    del rightVertex.edgeAliases[i]
                    del rightVertex.edgeJoinCols[i]
                    del rightVertex.outputJoinCols[i]
                    break
            circularEdge = self.findCycles()
            
    
    def addColumnSuffixes(self):
        for vertex in self.vertexDict.values():
            vertex.addColumnSuffixes(self.tableDict)
        self.hasSuffixes = True
    
    def getNeighborAliasesAndTables(self, alias):
        aliases = self.vertexDict[alias].edgeAliases
        tables = [self.tableDict[x] for x in aliases]
        return aliases, tables
        
    def printJoinGraph(self):
        if len(self.vertexDict) == 0:
            print("Join Graph Is Empty")
        counter = 0
        jIter = JoinQueryGraphIter(self)
        while jIter.hasNext():
            curVertex = jIter.getNextAsVertex()
            counter += 1
            predString = ""
            for pred in curVertex.predicates:
                predString +=  pred.toString() + " "
            print(curVertex.alias + " is the " + str(counter) + "th joined table with predicates: " + predString)
            for i in range(len(curVertex.edgeAliases)):
                print(curVertex.alias + " joins with " + curVertex.edgeAliases[i] + " on column " + curVertex.outputJoinCols[i])
    
    def getSQLQuery(self, countStar=False):
        query = ""
        if countStar:
            query += "SELECT COUNT(*) FROM "
        else:
            query += "SELECT * FROM "
        
        aliasNamePairs = list(self.tableDict.items())
        for i in range(len(aliasNamePairs) - 1):
            alias, tableName = aliasNamePairs[i]
            query += tableName + " AS " + alias + ", " 
        alias, tableName = aliasNamePairs[len(aliasNamePairs)-1]
        query += tableName + " AS " + alias + " "
        query += "WHERE "
        joinIter = JoinQueryGraphIter(self)
        curJoin = joinIter.getNext()
        curVertex = self.vertexDict[curJoin[2]]
        for pred in curVertex.predicates:
            query += curVertex.alias + "." + pred.toString() + " AND "
        while joinIter.hasNext():
            curJoin = joinIter.getNext()
            curVertex = self.vertexDict[curJoin[2]]
            for pred in curVertex.predicates:
                query += curVertex.alias + "." + pred.toString() + " AND "
            query += curJoin[0] + "." + curJoin[1] + "=" + curJoin[2] + "." + curJoin[3] + " AND "
        query = query[0:-4] + ";"
        return query
    
    def getSubQuery(self, aliases):
        subquery = JoinQueryGraph()
        for a in aliases:
            subquery.vertexDict[a] = self.vertexDict[a].copy()
        for a in aliases:
            edgeAliases = []
            edgeJoinCols = []
            outputJoinCols = []
            for i in range(len(subquery.vertexDict[a].edgeAliases)):
                if subquery.vertexDict[a].edgeAliases[i] in aliases:
                    edgeAliases.append(subquery.vertexDict[a].edgeAliases[i])
                    edgeJoinCols.append(subquery.vertexDict[a].edgeJoinCols[i])
                    outputJoinCols.append(subquery.vertexDict[a].outputJoinCols[i])
            subquery.vertexDict[a].edgeAliases = edgeAliases.copy()
            subquery.vertexDict[a].edgeJoinCols = edgeJoinCols.copy()
            subquery.vertexDict[a].outputJoinCols = outputJoinCols.copy()
            subquery.tableDict[a] = subquery.vertexDict[a].tableName
        
        subquery.equalityClasses = []
        for equalityClass in self.equalityClasses:
            newClass = []
            for member in equalityClass:
                alias = member.split(".")[0]
                if alias in aliases:
                    newClass.append(member)
            if len(newClass)>0:
                subquery.equalityClasses.append(newClass)
        
        for equalityClass, members in enumerate(subquery.equalityClasses):
            for member in members:
                subquery.equalityDict[member] = equalityClass
            
        return subquery
        
    def getIntermediateQueries(self):
        aliasList = list(self.vertexDict.keys())
        G = nx.Graph()
        G.add_nodes_from(aliasList)
        for n1, edges in self.edgeDict.items():
            for j in edges:
                n2 = j[0]
                G.add_edge(n1, n2)
        intermediateQueryAliases = []
        for nb_nodes in range(2, G.number_of_nodes()+1):
            for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G, nb_nodes)):
                if nx.is_connected(SG):
                    intermediateQueryAliases.append(list(SG.nodes))
        intermediateQueries = []
        for aliases in intermediateQueryAliases:
            intermediateQueries.append(self.getSubQuery(aliases))
        return intermediateQueries
    
    def copy(self):
        copyGraph = JoinQueryGraph()
        for k, v in self.vertexDict.items():
            copyGraph.vertexDict[k] = v.copy()
        for k, v in self.tableDict.items():
            copyGraph.tableDict[k] = v
        for k, v in self.equalityDict.items():
            copyGraph.equalityDict[k] = v
        for v in self.equalityClasses:
            copyGraph.equalityClasses.append([x for x in v])
        return copyGraph
        
    def getTopologicalGenerations(self, rootAlias):
        aliasToGeneration = dict()
        edgeQueue = deque()
        edgeQueue.append((rootAlias, 0))
        aliasesAlreadySeen = set([rootAlias])
        while not len(edgeQueue) == 0:
            curAlias, curGeneration = edgeQueue.pop()
            aliasToGeneration[curAlias] = curGeneration
            for childAlias in self.vertexDict[curAlias].edgeAliases:
                if childAlias not in aliasesAlreadySeen:
                    edgeQueue.append((childAlias, curGeneration + 1))
                    aliasesAlreadySeen.add(childAlias)
        return aliasToGeneration
            
    
# Breadth First Query Iterator
class JoinQueryGraphIter:
    
    def __init__(self, joinQueryGraph, rootAlias = None):
        self.joinQueryGraph = joinQueryGraph
        self.aliasesSeen = []
        self.queueOfJoins = deque()
        self.rootAlias = rootAlias
        if rootAlias == None:
            self.rootAlias = list(joinQueryGraph.vertexDict.keys())[0]
        self.queueOfJoins.append(["", "", self.rootAlias, ""])
        self.aliasesSeen.append(self.rootAlias)
    
    def hasNext(self):
        return not len(self.queueOfJoins) == 0
    
    def getNext(self):
        if len(self.queueOfJoins) == 0:
            return None
        curJoin = self.queueOfJoins.pop()
        curVertex = self.joinQueryGraph.vertexDict[curJoin[2]]
        for i in range(len(curVertex.edgeAliases)):
            child = curVertex.edgeAliases[i]
            if child not in self.aliasesSeen:
                self.queueOfJoins.append([curVertex.alias, curVertex.outputJoinCols[i], child, curVertex.edgeJoinCols[i]])
                self.aliasesSeen.append(child)
        return curJoin
    
    def getNextAsVertex(self):
        curJoin = self.getNext()
        return self.joinQueryGraph.vertexDict[curJoin[2]]
    
    def reset(self):
        self.queueOfJoins = deque
        rootAlias = list(self.joinQueryGraph.vertexDict.keys())[0]
        self.queueOfJoins.append(["", "", rootAlias, ""])
        self.aliasesSeen = []
            
class JoinHint:
    
    def __init__(self, tables, rowEstimate):
        self.tables = tables
        self.rowEstimate = rowEstimate