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
dirNames = ["SyntheticDataset1", "SyntheticDataset2", "MovieLens", "Flixster", "BookCrossing"]

verbose = False
generateRecommendations = False

rhos = [0.5, 1.0, 2.0]
losses = ["hinge", "square", "sigmoid", "logistic", "tanh"]
algs = ["SoftImpute", "WrMf"]
names = ["SoftImpute\t\t", "WRMF\t\t"]

for loss in losses: 
    if loss == "tanh": 
        for rho in rhos: 
            algs.append("MaxLocalAUC_loss=" + loss + "_rho=" + str(rho))
            names.append("MAUC " + (loss+ " " + "rho=" + str(rho)).ljust(10) + "\t") 
    else:         
        algs.append("MaxLocalAUC_loss=" + loss)
        names.append("MAUC " + loss.ljust(10) + "\t") 

for dirName in dirNames:
    resultsDir = PathDefaults.getOutputDir() + "ranking/" + dirName + "/"
    print("="*30 + dirName + "="*30)
    
    trainResultsTable = numpy.zeros((len(algs), len(ps)*4+2))
    testResultsTable = numpy.zeros((len(algs), len(ps)*4+2))
    figInd = 0    
    for s, alg in enumerate(algs): 
        resultsFileName = resultsDir + "Results" + alg + ".npz"
        try: 
            data = numpy.load(resultsFileName)
            trainMeasures, testMeasures, metaData, scoreInds = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
            
            trainResultsTable[s, 0:len(trainMeasures)] = trainMeasures
            testResultsTable[s, 0:len(testMeasures)] = testMeasures      
            
            if verbose: 
                logging.debug(names[s] + " metadata: " + str(metaData))
    
            if dirName == "MendeleyCoauthorsDoc" and generateRecommendations: 
                logging.debug("Generating recommendations for authors")
                authorIndexerFilename = PathDefaults.getDataDir() + "reference/authorIndexerDoc.pkl"
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
            
            if verbose: 
                logging.debug(meanMetrics)
                logging.debug(numpy.unravel_index(numpy.argmax(meanMetrics), meanMetrics.shape))
        except IOError: 
            logging.debug("Missing file " + modelSelectFileName)
    
    colNames = []
    for i, p in enumerate(ps): 
        colNames.append("p@" + str(p)) 
    for i, p in enumerate(ps): 
        colNames.append("r@" + str(p)) 
    #for i, p in enumerate(ps): 
    #    colNames.append("f1@" + str(p)) 
    #for i, p in enumerate(ps): 
    #    colNames.append("mrr@" + str(p)) 
    #colNames.extend(["localAUC@u", "AUC"])
    colNames.extend(["AUC"])
    
    #Restrict output to precision, recall and AUC 
    colInds = [0, 1, 2, 3, 4, 5, 13] 
    
    print("")
    print("-"*20 + "Train metrics" + "-"*20)
    print("\t" + Latex.listToRow(colNames))
    print(Latex.addRowNames(names, Latex.array2DToRows(trainResultsTable[:, colInds])))
    
    
    print("-"*20 + "Test metrics" + "-"*20)
    print("\t" +  Latex.listToRow(colNames))
    print(Latex.addRowNames(names, Latex.array2DToRows(testResultsTable[:, colInds])))
