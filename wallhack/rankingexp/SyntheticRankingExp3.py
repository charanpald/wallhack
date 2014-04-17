import numpy
import logging
import sys
import argparse 
import os
import errno
from wallhack.rankingexp.RankingExpHelper import RankingExpHelper
from sandbox.util.SparseUtils import SparseUtils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
m = 500
n = 200
k = 8 
u = 20.0/n
w = 1-u
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non zero elements: " + str(X.nnz))
logging.debug("Size of X: " + str(X.shape))

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = numpy.array([k])
defaultAlgoArgs.rhos = numpy.flipud(numpy.logspace(-7, -3, 5))
defaultAlgoArgs.lmbdasMlauc = 2.0**-numpy.arange(4, 12, 1)
#defaultAlgoArgs.lmbdasMlauc = numpy.array([0.1])
defaultAlgoArgs.maxIterations = m*20
defaultAlgoArgs.numRowSamples = 10
defaultAlgoArgs.numStepIterations = 500
defaultAlgoArgs.numAucSamples = 20
defaultAlgoArgs.initialAlg = "softimpute"
defaultAlgoArgs.recordStep = defaultAlgoArgs.numStepIterations
defaultAlgoArgs.rate = "optimal"
defaultAlgoArgs.alpha = 0.001
defaultAlgoArgs.t0 = 10**-3
defaultAlgoArgs.folds = 3
defaultAlgoArgs.nu = 50
defaultAlgoArgs.nuPrime = 50
defaultAlgoArgs.rho = 0.00
defaultAlgoArgs.ks = numpy.array([k])
defaultAlgoArgs.validationSize = 3
defaultAlgoArgs.u = u 


# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RankingExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "SyntheticDataset1"

# print args #
logging.info("Running on " + dataArgs.extendedDirName)
logging.info("Data params:")
keys = list(vars(dataArgs).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(dataArgs.__getattribute__(key)))

logging.info("Creating the exp-runner")
rankingExpHelper = RankingExpHelper(remainingArgs, defaultAlgoArgs, dataArgs.extendedDirName)
rankingExpHelper.printAlgoArgs()
#    os.makedirs(resultsDir, exist_ok=True) # for python 3.2
try:
    os.makedirs(rankingExpHelper.resultsDir)
except OSError as err:
    if err.errno != errno.EEXIST:
        raise

rankingExpHelper.runExperiment(X)