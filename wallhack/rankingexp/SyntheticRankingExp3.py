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
m = 1000
n = 500
k = 10 
u = 20.0/n
w = 1-u
X, U, s, V = SparseUtils.generateSparseBinaryMatrix((m,n), k, w, csarray=True, verbose=True, indsPerRow=200)
logging.debug("Number of non zero elements: " + str(X.nnz))
(m, n) = X.shape
logging.debug("Size of X: " + str(X.shape))
logging.debug("Number of non zeros: " + str(X.nnz))

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = 2**numpy.arange(3, 7)
defaultAlgoArgs.rhos = numpy.flipud(numpy.logspace(-7, -3, 5))
defaultAlgoArgs.folds = 4
defaultAlgoArgs.u = u
defaultAlgoArgs.maxIterations = 20*m
defaultAlgoArgs.numRowSamples = 20
defaultAlgoArgs.numStepIterations = 500
defaultAlgoArgs.numAucSamples = 20
defaultAlgoArgs.t0 = 10**-4 
defaultAlgoArgs.alpha = 0.1

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
logging.info("Running on SyntheticDataset1")
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