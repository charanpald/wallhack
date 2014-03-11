import numpy 
from sandbox.util.PathDefaults import PathDefaults
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
import logging 
import sys 
import pickle
import csv

numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#dirName = "SyntheticDataset1" 
dirName = "MendeleyCoauthors"
resultsDir = PathDefaults.getOutputDir() + "ranking/" + dirName + "/"
algs = ["SoftImpute", "WrMf"]
#algs = ["MaxLocalAUC"]

for alg in algs: 
    resultsFileName = resultsDir + "Results" + alg + ".npz"
    try: 
        
        data = numpy.load(resultsFileName)
        trainMeasures, testMeasures, metaData, scoreInds = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
        
        logging.debug(alg)
        logging.debug(trainMeasures)
        logging.debug(testMeasures)
        logging.debug(metaData)

        if dirName == "MendeleyCoauthors": 
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
        raise 
    
    modelSelectFileName = resultsDir + "ModelSelect" + alg + ".npz"
    """
    try: 
        data = numpy.load(modelSelectFileName)
        meanAucs, stdAucs = data["arr_0"], data["arr_1"]
        
        logging.debug(meanAucs)
        
        ks = numpy.array([10, 20, 50, 100])
        rhos = numpy.flipud(numpy.logspace(-4, -1, 5))         
        
        plt.contourf(rhos, ks, meanAucs)
        plt.colorbar()
        plt.show()
    except IOError: 
        logging.debug("Missing file " + modelSelectFileName)
    """