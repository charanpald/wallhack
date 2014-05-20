import numpy 
from sandbox.util.PathDefaults import PathDefaults
from sandbox.util.Latex import Latex
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
import logging 
import sys 
import pickle
import csv

numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

ps = [1, 3, 5]

dirName = "SyntheticDataset1" 
#dirName = "MovieLens" 
#dirName = "Flixster" 
#dirName = "MendeleyCoauthors"

generateRecommendations = False

resultsDir = PathDefaults.getOutputDir() + "ranking/" + dirName + "/"
algs = ["MaxLocalAUC", "SoftImpute", "WrMf", "Bpr"]
names = ["MLAUC",  "SoftImpute", "WrMf", "BPR"]

trainResultsTable = numpy.zeros((len(algs), len(ps)*2+2))
testResultsTable = numpy.zeros((len(algs), len(ps)*2+2))
figInd = 0

for s, alg in enumerate(algs): 
    resultsFileName = resultsDir + "Results" + alg + ".npz"
    try: 
        data = numpy.load(resultsFileName)
        trainMeasures, testMeasures, metaData, scoreInds = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
        
        trainResultsTable[s, 0:len(trainMeasures)] = trainMeasures
        testResultsTable[s, 0:len(testMeasures)] = testMeasures      
        
        logging.debug(alg)
        logging.debug("Metadata: " + str(metaData))

        if dirName == "MendeleyCoauthors" and generateRecommendations: 
            logging.debug("Generating recommendations for authors")
            authorIndexerFilename = PathDefaults.getDataDir() + "reference/authorIndexer.pkl"
            authorIndexerFile = open(authorIndexerFilename)
            authorIndexer = pickle.load(authorIndexerFile)
            authorIndexerFile.close()
            logging.debug("Loaded author indexer")
            
            reverseIndexer = authorIndexer.reverseTranslateDict()
            
            outputFileName = resultsDir + "Recommendations" + alg + ".csv"
            outputFile = open(outputFileName, "w")
            csvFile = csv.writer(outputFile, delimiter=',')
            
            for i in range(scoreInds.shape[0]):
                if i % 10000 == 0 : 
                    logging.debug("Iteration: " + str(i))
                    
                ids = [reverseIndexer[i]]                
                
                for j in range(scoreInds.shape[1]): 
                    ids.append(reverseIndexer[scoreInds[i, j]])
                
                csvFile.writerow(ids)
            outputFile.close()
            logging.debug("Wrote recommendations to " + outputFileName)
        
    except IOError: 
        logging.debug("Missing file " + resultsFileName)
        #raise 
    
    modelSelectFileName = resultsDir + "ModelSelect" + alg + ".npz"
    
    try: 
        data = numpy.load(modelSelectFileName)
        meanMetrics, stdMetrics = data["arr_0"], data["arr_1"]
        
        logging.debug(meanMetrics)
    except IOError: 
        logging.debug("Missing file " + modelSelectFileName)
    
colNames = []
for i, p in enumerate(ps): 
    colNames.append("p@" + str(p)) 
for i, p in enumerate(ps): 
    colNames.append("r@" + str(p)) 
colNames.extend(["localAUC@u", "AUC"])

print("")
print("-"*20 + "Train metrics" + "-"*20)
print(Latex.listToRow(colNames))
print(Latex.addRowNames(names, Latex.array2DToRows(trainResultsTable)))


print("-"*20 + "Test metrics" + "-"*20)
print(Latex.listToRow(colNames))
print(Latex.addRowNames(names, Latex.array2DToRows(testResultsTable)))
