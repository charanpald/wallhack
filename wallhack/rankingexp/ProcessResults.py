import numpy 
from sandbox.util.PathDefaults import PathDefaults
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
import logging 
import sys 

numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

dirName = "SyntheticDataset1" 
resultsDir = PathDefaults.getOutputDir() + "ranking/" + dirName + "/"
algs = ["WrMf", "MaxLocalAUC"]
algs = ["MaxLocalAUC"]

for alg in algs: 
    resultsFileName = resultsDir + "Results" + alg + ".npz"
    try: 
        
        data = numpy.load(resultsFileName)
        trainMeasures, testMeasures, metaData = data["arr_0"], data["arr_1"], data["arr_2"]
        
        logging.debug(alg)
        logging.debug(trainMeasures)
        logging.debug(testMeasures)
        logging.debug(metaData)
    except IOError: 
        logging.debug("Missing file " + resultsFileName)
    
    modelSelectFileName = resultsDir + "ModelSelect" + alg + ".npz"
    
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
