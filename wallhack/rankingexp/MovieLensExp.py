import numpy
import logging
import sys
import argparse 
import os
import errno
import sppy 
from wallhack.rankingexp.RankingExpHelper import RankingExpHelper
from sandbox.util.PathDefaults import PathDefaults

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)        
numpy.set_printoptions(precision=3, suppress=True, linewidth=150)

#Create a low rank matrix  
matrixFileName = PathDefaults.getDataDir() + "movielens/ml-100k/u.data" 
data = numpy.loadtxt(matrixFileName)
X = sppy.csarray((numpy.max(data[:, 0]), numpy.max(data[:, 1])), storagetype="row")
X[data[:, 0]-1, data[:, 1]-1] = numpy.array(data[:, 2]>3, numpy.int)
logging.debug("Read file: " + matrixFileName)
logging.debug("Shape of data: " + str(X.shape))
logging.debug("Number of non zeros " + str(X.nnz))

(m, n) = X.shape

# Arguments related to the dataset
dataArgs = argparse.Namespace()

# Arguments related to the algorithm
defaultAlgoArgs = argparse.Namespace()
defaultAlgoArgs.ks = 2**numpy.arange(3, 7)
defaultAlgoArgs.rhos = numpy.flipud(numpy.logspace(-7, -3, 5))
defaultAlgoArgs.folds = 4
defaultAlgoArgs.u = 20.0/m
defaultAlgoArgs.maxIterations = 10*m
defaultAlgoArgs.t0 = 10**-3
defaultAlgoArgs.alpha = 0.2
defaultAlgoArgs.initialAlg = "softimpute"

# data args parser #
dataParser = argparse.ArgumentParser(description="", add_help=False)
dataParser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
devNull, remainingArgs = dataParser.parse_known_args(namespace=dataArgs)
if dataArgs.help:
    helpParser  = argparse.ArgumentParser(description="", add_help=False, parents=[dataParser, RankingExpHelper.newAlgoParser(defaultAlgoArgs)])
    helpParser.print_help()
    exit()

dataArgs.extendedDirName = ""
dataArgs.extendedDirName += "MovieLens"

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