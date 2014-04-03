"""
Use the DBLP dataset to recommend experts. Find the optimal parameters. 
"""
import numpy 
import logging 
import sys 
import argparse
from wallhack.influence2.GraphRanker import GraphRanker 
from wallhack.influence2.RankAggregator import RankAggregator
from wallhack.influence2.ArnetMinerDataset import ArnetMinerDataset
from sandbox.util.Latex import Latex 
from sandbox.util.Evaluator import Evaluator
from sandbox.util.Util import Util
from sandbox.util.IdIndexer import IdIndexer
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=3, linewidth=160)
numpy.random.seed(21)

parser = argparse.ArgumentParser(description='Run reputation evaluation experiments')
parser.add_argument("-r", "--runLDA", action="store_true", help="Run Latent Dirchlet Allocation")
parser.add_argument("-d", "--useDocs", action="store_true", help="Use document database to find relevant authors")
args = parser.parse_args()

averagePrecisionN = 20 
ns = numpy.arange(5, 55, 5)
runLSI = not args.runLDA
knownAuthors = not args.useDocs

dataset = ArnetMinerDataset(runLSI=runLSI, knownAuthors=knownAuthors) 
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-100000.txt"
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-1000000.txt"
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-5000000.txt"
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-7000000.txt"
dataset.dataFilename = dataset.dataDir + "DBLP-citation-Feb21.txt" 
dataset.minDf = 10**-4
dataset.ks = [100, 200, 300, 400, 500, 600]
dataset.minDfs = [10**-3, 10**-4]
dataset.overwriteGraph = True
dataset.overwriteModel = True
dataset.overwriteVectoriser = True 

if not knownAuthors: 
    dataset.modelSelection()

#Sav the outputList and the graph
fich = open("//home//idexlab//" + "TotalMeasuresBM25" + ".txt", "w")
fgraph = open("//home//idexlab//" + "GraphTotal" + ".txt", "w")

for field in dataset.fields: 
    logging.debug("Field = " + field)
    if not knownAuthors: 
        dataset.learnModel() 
    dataset.overwriteVectoriser = False
    dataset.overwriteModel = False    
    
    if not knownAuthors: 
        relAuthorsDocSimilarity, relAuthorsDocCitations = dataset.findSimilarDocuments(field)
    else: 
        relAuthorsDocSimilarity, relAuthorsDocCitations = dataset.readKnownAuthors(field)
        #Relevant experts with BM25 score
        expertAuthors = dataset.readKnownAuthorsExpert(field)
	    

    relevantAuthors = set(relAuthorsDocSimilarity).union(set(relAuthorsDocCitations))
    logging.debug("Total number of relevant authors : " + str(len(relevantAuthors)))
    
    graph, authorIndexer = dataset.coauthorsGraph(field, relevantAuthors)
    #f = open(dataset.getCoauthorsFilename(field),'rb') #Save the graph of co-authors
    #storedlist=pickle.load(f)
    #for i in storedlist:
        #fgraph.write(str(i))
        #fgraph.write("\n"+ "Author vertices") 
        #fgraph.write(str(authorIndexer.getIdDict()))
        #fgraph.write(str(numpy.ones(graph.ecount())))
        #print(i)
    #fgraph.close()
    graph1, authorIndexer1 = dataset.GetIndexer(field, expertAuthors) #Get Index for the expert list BM25
    trainExpertMatches = dataset.matchExperts(relevantAuthors, dataset.trainExpertDict[field])   
    testExpertMatches = dataset.matchExperts(relevantAuthors, dataset.testExpertDict[field])     
    
    trainExpertMatchesInds = authorIndexer.translate(trainExpertMatches)
    testExpertMatchesInds = authorIndexer.translate(testExpertMatches) 
    relevantAuthorInds1 = authorIndexer.translate(relAuthorsDocSimilarity) 
    relevantAuthorInds2 = authorIndexer.translate(relAuthorsDocCitations) 
    relevantAuthorsInds = authorIndexer.translate(relevantAuthors)  
    expertAuthorsInds = authorIndexer1.translate(expertAuthors)#Get Ids our BM25 List
    
    assert (numpy.array(relevantAuthorInds1) < len(relevantAuthorsInds)).all()
    assert (numpy.array(relevantAuthorInds2) < len(relevantAuthorsInds)).all()
    
    if len(testExpertMatches) != 0:
        fich.write(field)
        fich.write("\n") 
        #First compute graph properties 
        computeInfluence = False
        #graphRanker = GraphRanker(k=100, numRuns=100, computeInfluence=computeInfluence, p=0.05, inputRanking=[relevantAuthorInds1, relevantAuthorInds2])
        graphRanker = GraphRanker(k=100, numRuns=100, computeInfluence=computeInfluence, p=0.05, inputRanking=None)
        outputLists = graphRanker.vertexRankings(graph, relevantAuthorsInds)
        for line in outputLists:
         ListeAuthorsFinale = authorIndexer.reverseTranslate(line)
         #fich.write(' '.join(map(str, ListeAuthorsFinale)))
         fich.write(str(ListeAuthorsFinale)+"\n")
        outputListsE = []
        #outputListsE.append(expertAuthorsInds)#Add BM25 Listto the outputList in order to aggregate scores later
        for line in outputListsE:
         ListeAuthorsFinale = authorIndexer1.reverseTranslate(line)
         #fich.write(' '.join(map(str, ListeAuthorsFinale)))
         fich.write(str(ListeAuthorsFinale)+"\n")
        #Save relevant authors 
        #numpy.save(dataset.dataDir  + "relevantAuthorsReputation" + field +  ".txt", outputLists)
        fich.write("-----------------------------------------------")  
          
        #for line in outputLists:  
         #fich.write(line[i]) 
        #Ajout du score de l'expertise
        #outputLists.append(expertAuthorsInds)

         
        itemList = RankAggregator.generateItemList(outputLists)
        methodNames = graphRanker.getNames()
        
        if runLSI: 
            outputFilename = dataset.getOutputFieldDir(field) + "outputListsLSI.npz"
        else: 
            outputFilename = dataset.getOutputFieldDir(field) + "outputListsLDA.npz"
            
        Util.savePickle([outputLists, trainExpertMatchesInds, testExpertMatchesInds], outputFilename, debug=True)
        
        numMethods = len(outputLists)
        precisions = numpy.zeros((len(ns), numMethods))
        averagePrecisions = numpy.zeros(numMethods)
        
        for i, n in enumerate(ns):     
            for j in range(len(outputLists)): 
                precisions[i, j] = Evaluator.precisionFromIndLists(testExpertMatchesInds, outputLists[j][0:n]) 
            
        for j in range(len(outputLists)):                 
            averagePrecisions[j] = Evaluator.averagePrecisionFromLists(testExpertMatchesInds, outputLists[j][0:averagePrecisionN], averagePrecisionN) 
        
        precisions2 = numpy.c_[numpy.array(ns), precisions]
        
        logging.debug(Latex.listToRow(methodNames))
        logging.debug("Computing Precision")
        logging.debug(Latex.array2DToRows(precisions2))
        logging.debug("Computing Average Precision")
        logging.debug(Latex.array1DToRow(averagePrecisions))
#fermer le fichier 
fich.close()

logging.debug("All done!")
