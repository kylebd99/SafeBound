# This is based off of simpleSQL.py in the pyparsing github examples file.
from JoinGraphUtils import  *
from pyparsing import (
    Word,
    delimitedList,
    Optional,
    Group,
    alphas,
    alphanums,
    Forward,
    oneOf,
    QuotedString,
    infixNotation,
    opAssoc,
    restOfLine,
    CaselessKeyword,
    ParserElement,
    pyparsing_common as ppc,
    removeQuotes,
)

def getSQLGrammar():
    ParserElement.enablePackrat()

    # define SQL tokens
    selectStmt = Forward()
    SELECT, FROM, AS, WHERE, AND, OR, IN, IS, NOT, NULL = map(
        CaselessKeyword, "SELECT FROM AS WHERE AND OR IN IS NOT NULL".split()
    )
    NOT_NULL = NOT + NULL

    ident = Word(alphas, alphanums + "_$").setName("identifier")
    ident.addParseAction(ppc.upcaseTokens)
    aliasAndColumnName = delimitedList(ident, '.')
    aliasAndColumnName.addParseAction(ppc.upcaseTokens)
    columnNameList = Group(delimitedList(aliasAndColumnName).setName("column_list"))
    alias = Group(ident + Optional(AS + ident))
    tableName = delimitedList(alias, ",", combine=False).setName("table name")
    tableNameList = Group(delimitedList(tableName).setName("table_list"))
    
    binop = oneOf("= != < > >= <= eq ne lt le gt ge LIKE", caseless=True).setName("binop")
    realNum = Group(ppc.real()).setResultsName("realNum")
    intNum = Group(ppc.signed_integer()).setResultsName('intNum')
    quoteStr = Group(QuotedString("'", unquoteResults=True))
    columnRval = (
        realNum | intNum | quoteStr | aliasAndColumnName
    ).setName("column_rvalue")  # need to add support for alg expressions
    
    whereCondition = Group(
        (aliasAndColumnName + binop + columnRval)
        | (aliasAndColumnName + IN + Group("(" + delimitedList(columnRval).setName("in_values_list") + ")"))
        | (aliasAndColumnName + IN + Group("[" + delimitedList(columnRval).setName("in_values_list") + "]"))
        | (aliasAndColumnName + IN + Group("(" + selectStmt + ")"))
        | (aliasAndColumnName + IS + (NULL | NOT_NULL))
        | (aliasAndColumnName + Group(NOT + binop) + columnRval)
    ).setName("where_condition")

    whereExpression = infixNotation(
        whereCondition,
        [
            (NOT, 1, opAssoc.RIGHT),
            (AND, 2, opAssoc.LEFT),
            (OR, 2, opAssoc.LEFT),
        ],
    ).setName("where_expression")

    # define the grammar
    selectStmt <<= (
        SELECT
        + ("*" | columnNameList)("columns")
        + FROM
        + tableNameList("tables")
        + Optional(Group(WHERE + whereExpression), "")("where")
    ).setName("select_statement")

    simpleSQL = delimitedList(Group(selectStmt), ";")

    # define Oracle comment format, and ignore them
    oracleSqlComment = "--" + restOfLine
    simpleSQL.ignore(oracleSqlComment)
    mySQLComment = "#" + restOfLine
    simpleSQL.ignore(mySQLComment)
    return simpleSQL

def SQLQueriesToJoinQueryGraphs(sqlQuery, verbose=False):
    SQLGrammar = getSQLGrammar()
    joinQueryGraphs = []
    SQLParseResult = SQLGrammar.parseString(sqlQuery)
    counter = 0
    for queryParse in SQLParseResult:
        if verbose:
            print("Parsing Query: " + str(counter))
        counter += 1
        queryGraph = JoinQueryGraph()
        aliases = [(x[0], x[2]) for x in queryParse['tables'] if len(x) >1 and x[1] == "AS"]
        for alias in aliases:
            queryGraph.addAlias(alias[0], alias[1])
        whereClausesAndConjunctions = queryParse['where'][0][1]
        for clause in whereClausesAndConjunctions:
            isConjunction = isinstance(clause, str)
            if not isConjunction:
                isJoin = (len(clause) == 5) and (clause[2] == "=")
                if isJoin:
                    queryGraph.addJoin(clause[0], clause[1], clause[3], clause[4])
                else:
                    predType = clause[2]
                    if clause[2][0] == "NOT":
                            predType = "NOT" + " " + str(clause[2][1])
                    if predType == 'IN':
                        if clause[3][1].getName() == 'intNum':
                            queryGraph.addPredicate(clause[0], clause[1], predType, [int(x[0]) for x in clause[3][1:-1]])
                        elif clause[3][1].getName() == 'realNum':
                            queryGraph.addPredicate(clause[0], clause[1], predType, [float(x[0]) for x in clause[3][1:-1]])
                        else:
                            queryGraph.addPredicate(clause[0], clause[1], predType, [str(x[0]) for x in clause[3][1:-1]])
                    elif predType.upper() == 'IS':
                        if clause[3] == "NOT":
                            queryGraph.addPredicate(clause[0], clause[1], "IS NOT NULL", None)
                        else:
                            queryGraph.addPredicate(clause[0], clause[1], "IS NULL", None)
                    elif clause[3].getName() == 'intNum':
                        queryGraph.addPredicate(clause[0], clause[1], predType, int(clause[3][0]))
                    elif clause[3].getName() == 'realNum':
                        queryGraph.addPredicate(clause[0], clause[1], predType, float(clause[3][0]))
                    else:
                        queryGraph.addPredicate(clause[0], clause[1], predType, str(clause[3][0]))
        queryGraph.buildJoinGraph()
        joinQueryGraphs.append(queryGraph)
    return joinQueryGraphs

def SQLFileToJoinQueryGraphs(fileAddress, verbose=False):
    file = open(fileAddress)
    fileContents = file.read()
    file.close()
    joinQueryGraphs = SQLQueriesToJoinQueryGraphs(fileContents, verbose)
    return joinQueryGraphs

def SQLFileToSQLStatements(fileAddress):
    file = open(fileAddress)
    fileContents = file.read()
    file.close()
    SQLStatements = fileContents.split(";")
    return SQLStatements
    
def ResultsFileToSizes(fileAddress):
    file = open(fileAddress)
    fileContents = file.read()
    file.close()
    lines = fileContents.split("\n")
    sizes = []
    for line in lines:
        values = line.split("#")
        sizes.append(int(values[-1]))
    return sizes
    