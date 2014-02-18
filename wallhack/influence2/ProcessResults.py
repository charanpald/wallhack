import sys 
import numpy 
import logging
from sandbox.util.IdIndexer import IdIndexer
from wallhack.influence2.ArnetMinerDataset import ArnetMinerDataset
from wallhack.influence2.GraphRanker import GraphRanker
from wallhack.influence2.RankAggregator import RankAggregator
from sandbox.util.Latex import Latex 
from sandbox.util.Util import Util 
from sandbox.util.Evaluator import Evaluator 

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

ranLSI = True
printOutputLists = False
printPrecisions = False 
printDocuments = True
numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
dataset = ArnetMinerDataset(runLSI=ranLSI)
#dataset.fields = ["Intelligent Agents"]

if printDocuments: 
    print("Reading article data")
    authorList, documentList, citationList = dataset.readAuthorsAndDocuments(useAbstract=False)
    print("Done")

ns = numpy.arange(5, 55, 5)
bestaverageTestPrecisions = numpy.zeros(len(dataset.fields))

computeInfluence = True
graphRanker = GraphRanker(k=100, numRuns=100, computeInfluence=computeInfluence, p=0.05, inputRanking=[1, 2])
methodNames = graphRanker.getNames()
methodNames.append("MC2")

numMethods = len(methodNames) 
averageTrainPrecisions = numpy.zeros((len(dataset.fields), len(ns), numMethods))
averageTestPrecisions = numpy.zeros((len(dataset.fields), len(ns), numMethods))

coverages = numpy.load(dataset.coverageFilename)
print("==== Coverages ====")
print(coverages)

for s, field in enumerate(dataset.fields): 
    if ranLSI: 
        outputFilename = dataset.getOutputFieldDir(field) + "outputListsLSI.npz"
        documentFilename = dataset.getOutputFieldDir(field) + "relevantDocsLSI.npy"
    else: 
        outputFilename = dataset.getOutputFieldDir(field) + "outputListsLDA.npz"
        documentFilename = dataset.getOutputFieldDir(field) + "relevantDocsLDA.npy"
        
    try: 
        print(field)  
        print("-----------")
        outputLists, trainExpertMatchesInds, testExpertMatchesInds = Util.loadPickle(outputFilename)
        
        graph, authorIndexer = Util.loadPickle(dataset.getCoauthorsFilename(field))

        trainPrecisions = numpy.zeros((len(ns), numMethods))
        testPrecisions = numpy.zeros((len(ns), numMethods))
        
        #Remove training experts from the output lists 
        trainOutputLists = []
        testOutputLists = [] 
        for outputList in outputLists:
            newTrainOutputList = []
            newTestOutputList = []
            for item in outputList: 
                if item not in testExpertMatchesInds: 
                    newTrainOutputList.append(item)
                if item not in trainExpertMatchesInds: 
                    newTestOutputList.append(item)
              
            trainOutputLists.append(newTrainOutputList)
            testOutputLists.append(newTestOutputList)
        
        for i, n in enumerate(ns):     
            for j, trainOutputList in enumerate(trainOutputLists): 
                testOutputList = testOutputLists[j]                
                
                trainPrecisions[i, j] = Evaluator.precisionFromIndLists(trainExpertMatchesInds, trainOutputList[0:n]) 
                testPrecisions[i, j] = Evaluator.precisionFromIndLists(testExpertMatchesInds, testOutputList[0:n]) 
                averageTrainPrecisions[s, i, j] = Evaluator.averagePrecisionFromLists(trainExpertMatchesInds, trainOutputList[0:n], n)
                averageTestPrecisions[s, i, j] = Evaluator.averagePrecisionFromLists(testExpertMatchesInds, testOutputList[0:n], n) 

        #Now look at rank aggregations
        relevantItems = set([])
        for trainOutputList in trainOutputLists: 
            relevantItems = relevantItems.union(trainOutputList)
        relevantItems = list(relevantItems)
        
        listInds = RankAggregator.greedyMC2(trainOutputLists, relevantItems, trainExpertMatchesInds, 20) 
        
        newOutputList = []
        for listInd in listInds: 
            newOutputList.append(testOutputLists[listInd])
        
        """
        newOutputList = []
        newOutputList.append(testOutputLists[0])
        newOutputList.append(testOutputLists[1])
        newOutputList.append(testOutputLists[2])
        newOutputList.append(testOutputLists[3])
        #newOutputList.append(testOutputLists[4])
        newOutputList.append(testOutputLists[5])
        #newOutputList.append(testOutputLists[6])
        """
        relevantItems = set([])
        for testOutputList in testOutputLists: 
            relevantItems = relevantItems.union(testOutputList)
        relevantItems = list(relevantItems)

        rankAggregate = RankAggregator.MC2(newOutputList, relevantItems)[0]
        j = len(outputLists)
        
        
        for i, n in enumerate(ns):
            testPrecisions[i, j] = Evaluator.precisionFromIndLists(testExpertMatchesInds, rankAggregate) 
            averageTestPrecisions[s, i, j] = Evaluator.averagePrecisionFromLists(testExpertMatchesInds, rankAggregate, n) 
        
        if printOutputLists: 
            print("Training expert matches")
            print(authorIndexer.reverseTranslate(trainExpertMatchesInds))
            
            print("\nTest expert matches")
            print(authorIndexer.reverseTranslate(testExpertMatchesInds))
            
            print("\nLists of experts")
            
            for ind in range(len(testOutputLists)):
                print(authorIndexer.reverseTranslate(testOutputLists[ind][0:50]))
                print("")
            print("-----------")
        
        if printDocuments: 
            print("Relevant Documents: ")
            relevantDocs = numpy.load(documentFilename)
            for i in relevantDocs: 
                print(documentList[i] + " " + ", ".join(authorList[i]))            
            print("-----------")
            
        if printPrecisions: 
            print("Precisions")
            print(trainPrecisions)
            print(averageTrainPrecisions[s, :, :])
            print(testPrecisions)
            print(averageTestPrecisions[s, :, :])
            print("-----------")
    except IOError as e: 
        print(e)

meanAverageTrainPrecisions = numpy.mean(averageTrainPrecisions, 0)
meanAverageTrainPrecisions = numpy.c_[numpy.array(ns), meanAverageTrainPrecisions]

meanAverageTestPrecisions = numpy.mean(averageTestPrecisions, 0)
meanAverageTestPrecisions = numpy.c_[numpy.array(ns), meanAverageTestPrecisions]

print("==== Summary ====")
print(Latex.listToRow(methodNames))
print(Latex.array2DToRows(meanAverageTrainPrecisions))

print(Latex.listToRow(methodNames))
print(Latex.array2DToRows(meanAverageTestPrecisions))

