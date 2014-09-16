import numpy 
import logging 
import pickle 
import csv
import sys 
from sandbox.recommendation.IterativeSoftImpute import IterativeSoftImpute
from sandbox.util.MCEvaluator import MCEvaluator 
from sandbox.util.PathDefaults import PathDefaults 
from wallhack.rankingexp.DatasetUtils import DatasetUtils 

"""
Use Mendeley author-documents and author-keywords to recommend contacts. 

"""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#go through: author, doc, different values of sigma, different algs, do model selection 

#First load the dataset 

dataset = "Keyword"
resultsDir = PathDefaults.getOutputDir() + "coauthors/"
X = DatasetUtils.mendeley(dataset=dataset)
X = X.toScipyCsc()

#Do some recommendation 
k = 64
learner = IterativeSoftImpute(k=k)

trainIterator = iter([X])
ZList = learner.learnModel(trainIterator)    
U, s, V = ZList.next()
U = U*s

U = numpy.ascontiguousarray(U)
V = numpy.ascontiguousarray(V)

maxItems = 5
orderedItems, scores = MCEvaluator.recommendAtk(U, V, maxItems, verbose=True)

#Now let's write out the similarities file 
logging.debug("Generating recommendations for authors")
authorIndexerFilename = PathDefaults.getDataDir() + "reference/authorIndexer" + dataset + ".pkl"
authorIndexerFile = open(authorIndexerFilename)
authorIndexer = pickle.load(authorIndexerFile)
authorIndexerFile.close()
logging.debug("Loaded author indexer")

reverseIndexer = authorIndexer.reverseTranslateDict()

outputFileName = resultsDir + "Recommendations.csv"
outputFile = open(outputFileName, "w")
csvFile = csv.writer(outputFile, delimiter='\t')

for i in range(orderedItems.shape[0]):
    if i % 10000 == 0 : 
        logging.debug("Iteration: " + str(i))
        
    row = [reverseIndexer[i]]                
    
    #Check author isn't recommended him/herself
    for j in range(orderedItems.shape[1]): 
        row = [reverseIndexer[i], reverseIndexer[orderedItems[i, j]], scores[i, j]]
    
        csvFile.writerow(row)
        
outputFile.close()
logging.debug("Wrote recommendations to " + outputFileName)

#Integrate marks code here 
